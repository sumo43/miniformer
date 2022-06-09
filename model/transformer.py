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

The code for transformer. attention mechanism in attention.py

"""

class TransformerFC(Module):
    def __init__(self, mp):
        super().__init__()
        self.d_model = mp.d_model
        self.d_ff = mp.d_ff

        self.fc1 = torch.nn.Linear(self.d_model, self.d_ff)
        self.fc2 = torch.nn.Linear(self.d_ff, self.d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) 
        x = self.fc2(x)

        return x

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
        self.ff = TransformerFC(mp)

    def forward(self, x):
        assert x.shape[-1] == self.d_model

        x_prev = x
        x = self.mha(*self.expand(x))
        x = self.add_norm(x_prev, x)
        x_prev = x
        x = self.ff(x)
        x = self.add_norm(x_prev, x)
        
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
        self.n_encoders = mp.n_encoders
        self.device = mp.device

        self.encoder = nn.Sequential(
            *[EncoderBlock(self.mp).to(self.device) for i in range(self.n_encoders)])
        
    def forward(self, x):
        return self.encoder(x)

class TransformerEmbedding(Module):
    def __init__(self, mp, size):
        super().__init__()

        self.d_model = mp.d_model 
        self.embedding = torch.nn.Embedding(size, self.d_model, padding_idx=1)

    def forward(self, x):

        x = self.embedding(x)
        return x


class DecoderBlock(Module):
    
    def __init__(self, mp):

        super().__init__()

        self.d_model = mp.d_model
        self.mha_2 = MultiHeadAttention(mp, masked=True)
        self.mha_2 = MultiHeadAttention(mp)
        self.add_norm = AddNorm(mp)
        self.ff = TransformerFC(mp)

    def forward(self, inputs):

        x, x_e = inputs

        # x_e is the encoder output, which we feed back in as a query and key at each layer

        assert x_e.shape[-1] == self.d_model
        assert x.shape[-1] == self.d_model

        x_prev = x
        x = self.mha_2(*self.expand(x))
        x = self.add_norm(x_prev, x)
        x_prev = x
        Q = x
        x = self.mha_2(*self.expand_VK(x_e), Q)
        x = self.add_norm(x_prev, x)
        x_prev = x
        x = self.ff(x)
        x = self.add_norm(x_prev, x)
        
        assert x.shape[-1] == self.d_model

        return (x, x_e)
    
    def expand_VK(self, x):

        V = x.clone()
        K = x.clone()

        return (V, K)
    
    def expand(self, x):
        Q = x.clone()
        K = x.clone()
        V = x.clone()

        return (V, K, Q)

class TransformerDecoder(Module):
    def __init__(self, mp):

        super().__init__()

        self.mp = mp
        self.h = mp.h
        self.d_model = mp.d_model
        self.es_vocab_size = mp.es_vocab_size
        self.n_decoders = mp.n_decoders

        self.decoder = nn.Sequential(
            *[DecoderBlock(self.mp) for i in range(self.n_decoders)])
        
        
    def forward(self, x):

        assert type(x) == tuple

        x = self.decoder(x)

        return x

class Transformer(Module):

    def __init__(self, mp):

        super().__init__()
        
        self.mp = mp
        self.en_vocab_size = mp.en_vocab_size
        self.es_vocab_size = mp.es_vocab_size
        self.d_model = mp.d_model
        self.d_ff = mp.d_ff

        self.input_embedding = TransformerEmbedding(self.mp, self.en_vocab_size + self.es_vocab_size)
        self.output_embedding = TransformerEmbedding(self.mp, self.es_vocab_size)
        self.pos_encoding = PositionalEncoder(self.mp)
        self.encoder = TransformerEncoder(self.mp)
        self.decoder = TransformerDecoder(self.mp)
        self.device = mp.device

        # dont need softmax because were using crossentropy loss (includes it)
        self.head = nn.Sequential(
            TransformerFC(self.mp),
            torch.nn.Linear(self.d_model, self.es_vocab_size)
        )  
    
    def forward(self, _input, _output):

        # the input is a 1-d token vector
        assert len(_input.shape) == 2
        assert len(_output.shape) == 2

        _input = self.input_embedding(_input)
        _output = self.output_embedding(_output)

        _input = self.pos_encoding(_input)
        _output = self.pos_encoding(_output)

        encoder_output = self.encoder(_input)

        _output, _encoder_output = self.decoder((_output, encoder_output))

        _output = self.head(_output)

        return _output