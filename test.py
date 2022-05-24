from utils.general import ModelParams
from model.attention import SDPAttention, MultiHeadAttention
from model.transformer import EncoderBlock, TransformerEncoder, DecoderBlock, TransformerDecoder, Transformer
from utils.data import load_data
from utils.preproc import preprocess_data, postprocess_data, to_string
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

train, val =  load_data(mp)
in_ds, out_ds, eng_vocab, sp_vocab = preprocess_data(mp, train, val, test=True)

d = TransformerDecoder(mp)

print(d(inputs).shape)

#megatron = Transformer(n_dim, n_heads, ...)

t = Transformer(mp)
x = t(in_ds[0], out_ds[0])
print(x.shape)

x = postprocess_data(mp, x)
print(to_string(x))


print('those tests are all we need')