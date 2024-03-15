import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# x = torch.arange(0, 12).view(1, 3, 4)
# print(x)
# print(x.transpose(-2, -1))
# print(x.transpose(0, 1).size())


class MultiHead(nn.Module):
    def __init__(self, d_model=512, num_head=8, mask=None,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert d_model % num_head == 0, "d_model divisible by num_head"
        self.d_model = d_model
        self.num_head = num_head
        self.mask = mask
        self.dim_h = d_model // num_head

        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def split_(self, x):
        return x.view(x.size(0), x.size(1) * self.num_head, self.dim_h).transpose(0, 1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        seq_len, btz, _ = query.size()
        # Linear + split
        q, k, v = self.split_(self.linear_q(query)), self.split_(self.linear_k(
            key)), self.split_(self.linear_v(value))

        _, _, d_key = k.size()
        query_scaled = q / math.sqrt(d_key)

        attn_output_weights = torch.bmm(query_scaled, k.transpose(-2, -1))

        if self.mask:
            attn_output_weights.masked_fill_(self.mask == 0, -1e12)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        atten_output = torch.bmm(attn_output_weights, v)

        # concat
        atten_output = atten_output.transpose(
            0, 1).contiguous().view(btz * seq_len, self.d_model)
        atten_output = self.linear_out(atten_output)
        return atten_output.view(seq_len, btz, self.d_model)


class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self, d_model, d_ff, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        step_compnent = torch.arange(0, d_model, 2).float()
        w = torch.exp(step_compnent * (-math.log(10_000) / d_model))

        pe[:, 0, 0::2] = torch.sin(w * position)
        pe[:, 0, 1::2] = torch.cos(w * position)

        self.register_buffer("pe", pe)

    def forward(self, x):

        # batch first false
        return x + self.pe[:x.size(0)]


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, mask=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multi_head_atten = MultiHead(d_model, num_heads, mask)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = PositionWiseFeedForwardNetworks(d_model, d_ff, self.dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        atten_output = self.multi_head_atten(x, x, x)
        # add & Norm
        x = self.layer_norm1((x + atten_output))
        x = self.dropout(x)
        ff_output = self.ff(x)
        x = self.layer_norm2((x + ff_output))
        return x


class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, src_mask=None, trg_mask=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multi_head_self_atten = MultiHead(d_model, num_heads, src_mask)
        self.multi_head_cross_atten = MultiHead(d_model, num_heads, trg_mask)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = PositionWiseFeedForwardNetworks(d_model, d_ff, self.dropout)

    def forward(self, trg: torch.Tensor, src: torch.Tensor, ):
        pass


class Transformer(nn.Module):
    def __init__(self, d_model: int, src_embed_size: int,  trg_embed_size: int, num_head: int = 8, num_layers: int = 6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_enc = nn.Embedding(src_embed_size, d_model)
        self.embed_dec = nn.Embedding(trg_embed_size, d_model)

        # TODO: complete!
    def generate_mask(self):
        self


# model = MultiHead()
# x = torch.randn(40, 32, 512)
# print(model(x, x, x).size())

# linear = nn.Linear(10, 10)
# x = torch.randn(200)
# y = x.view(2, 10, 10)
# print(linear(y))
# z = x.view(20, 10)
# print(linear(z).view(2, 10, 10))


#
# model = EncoderBlock(512, 8, 2048, dropout=0.2)
# x = torch.randn(40, 32, 512)
# print(model(x).size())
seq_len = 40
d_model = 512
x = torch.randn(seq_len, 32, d_model)
pe = PositionalEncoding(100, d_model)
print(pe(x))
