from utils.general import ModelParams
from model.attention import SDPAttention, MultiHeadAttention
from model.transformer import EncoderBlock, TransformerEncoder, DecoderBlock, TransformerDecoder, Transformer
from utils.data import load_data
from utils.preproc import preprocess_data
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

train, val =  load_data(mp)
eng, sp = preprocess_data(mp, train, val, test=True)

t = Transformer(mp)

test_t = torch.zeros((1, 1, 512)).type(torch.IntTensor)
x = t(test_t)

print('those tests are all we need')
