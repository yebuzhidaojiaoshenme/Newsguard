import math
import torch
import torch.nn as nn
from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        return self.sigmoid(h) * mask_cls.float()


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, step][:, None, :] if step else self.pe[:, :emb.size(1)]
        return self.dropout(emb)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        input_norm = self.layer_norm(inputs) if iter != 0 else inputs
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_inter_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        pos_emb = self.pos_emb.pe[:, :top_vecs.size(1)]
        x = top_vecs * mask[:, :, None].float() + pos_emb
        for i in range(len(self.transformer_inter)):
            x = self.transformer_inter[i](i, x, x, 1 - mask)
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x)).squeeze(-1) * mask.float()
        return sent_scores
