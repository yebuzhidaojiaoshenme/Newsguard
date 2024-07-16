import torch
import torch.nn as nn
import numpy as np
from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState

MAX_SIZE = 5000

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('mask', self._get_attn_subsequent_mask(MAX_SIZE))

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=None, layer_cache=None, step=None):
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = torch.cat((previous_input, input_norm), dim=1) if previous_input is not None else input_norm
        query = self.self_attn(all_input, all_input, input_norm, mask=dec_mask, layer_cache=layer_cache, type="self")
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask, layer_cache=layer_cache, type="context")
        output = self.feed_forward(self.drop(mid) + query)
        return output, all_input

    def _get_attn_subsequent_mask(self, size):
        subsequent_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList([TransformerDecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, memory_lengths=None, step=None, cache=None, memory_masks=None):
        emb = self.embeddings(tgt)
        output = self.pos_emb(emb, step)
        src_pad_mask = memory_masks.expand(state.src.size(0), tgt.size(1), memory_masks.size(-1)) if memory_masks is not None else state.src.data.eq(self.embeddings.padding_idx).unsqueeze(1).expand(state.src.size(0), tgt.size(1), state.src.size(1))
        tgt_pad_mask = tgt.data.eq(self.embeddings.padding_idx).unsqueeze(1).expand(tgt.size(0), tgt.size(1), tgt.size(1))
        saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = state.previous_layer_inputs[i] if state.previous_input is not None else None
            output, all_input = self.transformer_layers[i](output, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=prev_layer_input, layer_cache=state.cache["layer_{}".format(i)] if state.cache is not None else None, step=step)
            saved_inputs.append(all_input)

        output = self.layer_norm(output)
        state = state.update_state(tgt, torch.stack(saved_inputs))
        return output, state

    def init_decoder_state(self, src, memory_bank, with_cache=False):
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state

class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {"layer_{}".format(l): {"self_keys": None, "self_values": None, "memory_keys": None, "memory_values": None} for l in range(num_layers)}

    def repeat_beam_size_times(self, beam_size):
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        self.src = fn(self.src, 0)
        if self.cache is not None:
            for k, v in self.cache.items():
                if v is not None:
                    if isinstance(v, dict):
                        self._recursive_map(v)
                    else:
                        self.cache[k] = fn(v, 0)

    def _recursive_map(self, struct, batch_dim=0):
        for k, v in struct.items():
            if v is not None:
                if isinstance(v, dict):
                    self._recursive_map(v)
                else:
                    struct[k] = fn(v, batch_dim)
