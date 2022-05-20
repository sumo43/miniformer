from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import time

from model.attention import MultiHeadAttention

class AddNorm(Module):
    def __init__(self, mp):
        super().__init__()

        self.d_model = mp.d_model
        self.norm = torch.nn.LayerNorm(self.d_model)

    def forward(self, x_prev, x):
        x = x_prev + x
        x = self.norm(x)

        return x


class EncoderBlock(Module):
    def __init__(self, mp):
        super().__init__()

        self.d_model = mp.d_model
        self.mha = MultiHeadAttention(mp)
        self.add_norm = AddNorm(mp)
        self.ff = torch.nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        assert x.shape[1] == self.d_model

        x_prev = x
        x = self.mha(*self.expand(x))
        x = self.add_norm(x_prev, x)
        x_prev = x
        x = self.ff(x)
        x = self.add_norm(x_prev, x)
        
        assert x.shape[1] == self.d_model
        return x
    
    def expand(self, x):
        Q = x.clone()
        K = x.clone()
        V = x.clone()

        return (Q, K, V)


class TransformerEncoder(Module):
    def __init__(self, mp):

        self.d_model = mp.d_model
        self.h = mp.h

        self.encoder = nn.Sequential(
            [EncoderBlock(n_d, h) for i in range(n_l)])

    def forward(self, x):
        return self.encoder(x)


class TransformerEmbedding:
    def __init__(self, n_d=512):
        pass

    def forward(self, x):
        return x


"""

TransformerBlock:
TransformerEmbedding -> TransformerEncoder, which is just a list of EncoderBlock
class TransformerBlock
"""


class DecoderBlock(Module):

    def __init__(self, n_d=512, n_heads=8):

        pass


class TransformerDecoder(Module):
    def __init__(self, n_d=512, n_heads=8, n_l=6):

        self.decoder = nn.Sequential(
            [DecoderBlock(n_d, n_heads) for i in range(n_l)])

        pass

    def forward(self, x):
        return self.decoder(x)


class Transformer:

    def __init__(self, n_d=512, n_heads=8, n_l=6):
        pass
