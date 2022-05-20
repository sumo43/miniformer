from utils.general import ModelParams
from model.attention import SDPAttention, MultiHeadAttention
from model.transformer import EncoderBlock, TransformerEncoder, DecoderBlock, TransformerDecoder
import torch

# default params
mp = ModelParams()

sdpa = SDPAttention(mp)

# fake qkv

Q = torch.randn((1, mp.d_k))
K = torch.randn((1, mp.d_k))
V = torch.randn((1, mp.d_v))

sdpa(Q, K, V)

mha = MultiHeadAttention(mp)

Q = torch.randn((1, mp.d_model))
K = torch.randn((1, mp.d_model))
V = torch.randn((1, mp.d_model))

mha(Q, K, V)

eb = EncoderBlock(mp)
print(eb(Q).shape)

e = TransformerEncoder(mp)
print(e(Q).shape)

db = DecoderBlock(mp)

inputs = (Q, K)

print(db(inputs)[1].shape)

d = TransformerDecoder(mp)

print(d(inputs)[1].shape)



#megatron = Transformer(n_dim, n_heads, ...)

print('those tests are all we need')
