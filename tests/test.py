import torch
from miniformer.model import Config, Transformer

"""
Tests to make sure shapes and stuff are correct
These probably won't catch architecture bugs that may prevent it from converging
"""

# default params
config = Config()
t = Transformer(config)

test_x = torch.ones((1, 5)).type(torch.IntTensor)

"""

sdpa = SDPAttention(mp)

# fake qkv

Q = torch.randn((1, 1, mp.d_k))
K = torch.randn((1, 1, mp.d_k))
V = torch.randn((1, 1, mp.d_v))

sdpa(Q, K, V)

mha = MultiHeadAttention(mp)

Q = torch.randn((1, 1, mp.d_model))
K = torch.randn((1, 1, mp.d_model))
V = torch.randn((1, 1, mp.d_model))

mha(Q, K, V)

eb = EncoderBlock(mp)
print(eb(Q).shape)

e = TransformerEncoder(mp)
print(e(Q).shape)

db = DecoderBlock(mp)

inputs = (Q, K)

print(db(inputs)[0][1].shape)

inputs, outputs = load_data(mp)
in_ds, out_ds, eng_vocab, sp_vocab = preprocess_data(mp, inputs, outputs, test=True)

#d = TransformerDecoder(mp)

#print(d(inputs).shape)

#megatron = Transformer(n_dim, n_heads, ...)

dataset = zip(inputs, outputs)

config = ModelConfig()

t = Transformer(mp)
x = t(in_ds[0], out_ds[0])
print(x.shape)

x = postprocess_data(mp, x)
print(to_string(x, sp_vocab))

x = torch.ones((4, 4), dtype=torch.float)

def mask(x):
    tri_mask = torch.triu(torch.ones(size=x.shape, dtype=torch.float)) * VERY_SMOL_NUM
    print(tri_mask)
    x = x + tri_mask
    return x

tr = TransformerTrainer(mp, t)

print('those tests are all we need')
"""
