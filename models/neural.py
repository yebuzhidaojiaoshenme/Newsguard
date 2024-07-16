import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type="dot"):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], "Invalid attention type."
        self.attn_type = attn_type

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

    def score(self, h_t, h_s):
        if self.attn_type == "general":
            h_t = self.linear_in(h_t.view(-1, h_t.size(-1))).view(h_t.size())
        if self.attn_type in ["general", "dot"]:
            return torch.bmm(h_t, h_s.transpose(1, 2))
        else:
            wq = self.linear_query(h_t.view(-1, self.dim)).view(h_t.size(0), h_t.size(1), 1, self.dim)
            uh = self.linear_context(h_s.contiguous().view(-1, self.dim)).view(h_s.size(0), 1, h_s.size(1), self.dim)
            wquh = torch.tanh(wq + uh)
            return self.v(wquh.view(-1, self.dim)).view(h_t.size(0), h_t.size(1), h_s.size(1))

    def forward(self, source, memory_bank, memory_lengths=None, memory_masks=None):
        one_step = source.dim() == 2
        if one_step:
            source = source.unsqueeze(1)

        align = self.score(source, memory_bank)

        if memory_masks is not None:
            align.masked_fill_(~memory_masks.byte(), -float('inf'))

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1)).unsqueeze(1)
            align.masked_fill_(~mask, -float('inf'))

        align_vectors = F.softmax(align.view(-1, align.size(-1)), -1).view(align.size())

        c = torch.bmm(align_vectors, memory_bank)
        concat_c = torch.cat([c, source], -1).view(-1, 2 * self.dim)
        attn_h = self.linear_out(concat_c).view(source.size(0), -1, self.dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            return attn_h.squeeze(1), align_vectors.squeeze(1)
        return attn_h.transpose(0, 1).contiguous(), align_vectors.transpose(0, 1).contiguous()

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        super(MultiHeadedAttention, self).__init__()
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        def shape(x):
            return x.view(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count * self.dim_per_head)

        key = shape(self.linear_keys(key))
        value = shape(self.linear_values(value))
        query = shape(self.linear_query(query))

        query = query / math.sqrt(self.dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).expand_as(scores), -1e18)

        attn = self.dropout(self.softmax(scores))
        context = torch.matmul(attn, value)
        context = unshape(context)
        return self.final_linear(context)

class DecoderState(object):
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()