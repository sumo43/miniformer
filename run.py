from utils.general import ModelParams
from model.attention import SDPAttention
import torch

# default params
mp = ModelParams()

sdpa = SDPAttention(mp)

# fake qkv

Q = torch.randn((1, mp.d_k))
K = torch.randn((1, mp.d_k))
V = torch.randn((1, mp.d_v))

sdpa(Q, K, V)


#megatron = Transformer(n_dim, n_heads, ...)
