from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import time

from model.attention import MultiHeadAttention
from utils.general import PositionalEncoder
from einops.einops import rearrange

"""

Notes in attention.py.

"""

class ModelConfig():

    # model parameters
    m = None # width of input sequence
    k = 64 # key dimension size
    v = 64 # value dimension size
    d = 512 # dimension of hidden state between blocks
    h = 8 # number of heads
    b = 1 # batch dimension
    d_ff = 1024 # size of fully-connected layer

    # hyperparameters for training
    p_drop = 0.1
    lr = 1e-6

    # other parameters
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.embedding_size = 1024
    self.vocab_size = 30000 # shared vocab, if doing NLP

class FeedForward(Module):
    def __init__(self, mp):
        super().__init__()

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

class MQADecoder(torch.nn.Module):

    """ 
    Multi Query Attention: Like the above, but only the query has heads. 
    P_v and P_k are shared across all the heads 

    Params:
    m: length of the input to the decoder
    k: key dimension
    v: value dimension
    d: embedding dimension (d_model)
    h: number of heads
    b: batch dimension

    Args:
    x: a vector with shape [b, d]. Input from previous decoder step
    K_prev: a matrix with shape [b, m, k]. Is initially the encoder output.
    V_prev: a matrix with the shape [b, m, v]. Is initially the encoder output.
    P_q: a tensor with shape [h, d, k]. model weights.
    P_k: a tensor with shape [h, d, k]. model weights.
    P_v: a tensor with shape [h, d, v]. model weights.
    P_o: a tensor with shape [h, d, v]. model weights.

    Returns : 
    y: a vector with shape [b, d]
    K_new: a matrix with shape [b, m+1, k]
    V_new: a matrix with the shape [b, m+a, v]
    """

    def __init__(self):
        # same thing as torch.nn.Linear. This enables the use of einops
        self.P_q = torch.kaiming_uniform_(torch.empty(h, d, k))
        self.P_k = torch.kaiming_uniform_(torch.empty(h, d, k))
        self.P_v = torch.kaiming_uniform_(torch.empty(h, d, v))
        self.P_o = torch.kaiming_uniform_(torch.empty(h, d, v))

    def forward(x, K_prev, V_prev):

        # compute q, K, V. Concat K and V
        q = torch.einsum('d, hdk -> hk', x, self.P_q)
        K_new = torch.concat([K_prev, torch.expand_dims(torch.einsum('bd, dk -> bk', M, self.P_k), 2)], 2)
        V_new = torch.concat([V_prev, torch.expand_dims(torch.einsum('bd, dv -> bv', M, self.P_v), 2)], 2)

        # perform attention along dimension h (heads)
        logits = torch.einsum('hk, bmk -> bhm', q, K_new)
        key_weights = torch.nn.Softmax(logits)
        o = torch.einsum('bmv, hm -> bhv', key_weights, V_new)

        # matmul with our linear layers at the end 
        y = torch.einsum('bhv, hdv -> bd', o, P_o)

        return y, K_new, V_new


class MHAEncoder(torch.nn.Module):
    
    """ 
    Multi Head Attention for the Encoder. 

    Params:
    m: length of the input to the decoder
    k: key dimension
    v: value dimension
    d: embedding dimension (d_model)
    h: number of heads
    b: batch dimension

    Args:
    x: a vector with shape [b d]. This becomes Q, K, V
    P_q: a tensor with shape [h, d, k]. model weights.
    P_k: a tensor with shape [h, d, k]. model weights.
    P_v: a tensor with shape [h, d, v]. model weights.
    P_o: a tensor with shape [h, d, v]. model weights.

    Returns : 
    y: a vector with shape [b, d]
    """

    m = None
    k = 64
    v = 64
    d = 512
    h = 8
    b = 1

    def __init__(self):
        self.P_q = torch.xavier_uniform_(torch.empty(h, d, k))
        self.P_k = torch.xavier_uniform_(torch.empty(h, d, k))
        self.P_v = torch.xavier_uniform_(torch.empty(h, d, v))
        self.P_o = torch.xaview_uniform_(torch.empty(h, d, v))
    
    def forward(x):

        # linear layers before attention
        Q = torch.einsum('bd, hdk -> bhk', x, P_q)
        K = torch.einsum('bd, hdk -> bhk', x, P_k)
        V = torch.einsum('bd, hdv -> bhv', x, P_v)

        # attention
        logits = torch.einsum('bhk, bhk -> bh', Q, K)
        key_weights = torch.nn.Softmax(logits)
        o = torch.einsum('bh, bhv -> bhv', key_weights, V)

        # linear layer after attention
        y = torch.einsum('bhv, hdv -> bd', o, P_o)

        return y

class EncoderBlock(Module):
    def __init__(self, mp):
        super().__init__()
        self.mha = MHAEncoder()
        self.add_norm = AddNorm()
        self.ff = FeedForward()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x_prev = x
        x = self.mha(x)
        x = self.dropout(x) # dropout after attention
        x = self.add_norm(x_prev, x)
        x_prev = x
        x = self.ff(x)
        x = self.dropout(x)
        x = self.add_norm(x_prev, x)
                
        return x

class DecoderBlock(Module):
    
    def __init__(self, mp):

        super().__init__()

        self.d_model = mp.d_model
        self.mha_1 = MultiHeadAttention(mp, masked=True)
        self.mha_2 = MultiHeadAttention(mp)
        self.add_norm = AddNorm(mp)
        self.ff = TransformerFC(mp)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, inputs):

        x, K, V = inputs # x_e is the encoder output, which we feed back in as a query and key at each layer

        x_prev = x
        x, K, V = self.mha_1(x)
        x = self.add_norm(x_prev, x)
        x = self.dropout(x)
        x_prev = x
        Q = x
        x, K, V = self.mha_2(*self.expand_VK(x_e), Q)
        x = self.add_norm(x_prev, x)
        x = self.dropout(x)
        x_prev = x
        x = self.ff(x)
        x = self.add_norm(x_prev, x)
        x = self.dropout(x)
        
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

        #torch.nn.Embedding(size, self.d_model, padding_idx=3)

        self.input_embedding = TransformerEmbedding(self.mp, self.en_vocab_size)
        self.output_embedding = TransformerEmbedding(self.mp, self.en_vocab_size)
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

        _input = self.input_embedding(_input)
        _input = self.pos_encoding(_input)
        encoder_output = self.encoder(_input)
        """

        _output = self.output_embedding(_output)

        _input = self.pos_encoding(_input)
        _output = self.pos_encoding(_output)

        encoder_output = self.encoder(_input)

        _output, _encoder_output = self.decoder((_output, encoder_output))

        _output = self.head(_output)
        """

        return _input