from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import time

from model.attention import MultiHeadAttention
from utils.general import PositionalEncoder

"""
TransformerBlock:
TransformerEmbedding -> TransformerEncoder, which is just a list of EncoderBlock
class TransformerBlock

TODO change the linear layers to "expand"
"""


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
        
        super().__init__()
        self.mp = mp
        self.d_model = mp.d_model
        self.h = mp.h

        self.encoder = nn.Sequential(
            *[EncoderBlock(self.mp) for i in range(self.h)])

    def forward(self, x):
        return self.encoder(x)

class TransformerEmbedding:
    def __init__(self, n_d=512):
        pass

    def forward(self, x):
        return x


class DecoderBlock(Module):
    
    def __init__(self, mp):

        super().__init__()

        self.d_model = mp.d_model
        self.mha_1 = MultiHeadAttention(mp, masked=True)
        self.mha_2 = MultiHeadAttention(mp)
        self.add_norm = AddNorm(mp)
        self.ff = torch.nn.Linear(self.d_model, self.d_model)
    
        pass

    def forward(self, inputs):

        x, x_e = inputs

        # x_e is the encoder output, which we feed back in as a query and key at each layer


        assert x_e.shape[1] == self.d_model
        assert x.shape[1] == self.d_model

        x_prev = x
        x = self.mha_1(*self.expand(x))
        x = self.add_norm(x_prev, x)
        x_prev = x
        V = x
        x = self.mha_2(*self.expand_QK(x_e), V)
        x = self.add_norm(x_prev, x)
        x_prev = x
        x = self.ff(x)
        x = self.add_norm(x_prev, x)
        
        assert x.shape[1] == self.d_model

        return (x, x_e)
    
    def expand_QK(self, x):

        Q = x.clone()
        K = x.clone()

        return (Q, K)
    
    def expand(self, x):
        Q = x.clone()
        K = x.clone()
        V = x.clone()

        return (Q, K, V)

class TransformerDecoder(Module):
    def __init__(self, mp):

        super().__init__()

        self.mp = mp
        self.h = mp.h

        self.decoder = nn.Sequential(
            *[DecoderBlock(self.mp) for i in range(self.h)])
        
    def forward(self, x):

        assert type(x) == tuple

        return self.decoder(x)

class Transformer:

    def __init__(self, mp):
        
        self.mp = mp

        #self.embedding = TransformerEmbedding(self.mp)
        self.pos_encoding = PositionalEncoder(self.mp)
        self.encoder = TransformerEncoder(self.mp)
        self.decoder = TransformerDecoder(self.mp)
    
    def forward(self, x):

        return x