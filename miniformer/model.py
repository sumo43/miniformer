from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import time
from einops.einops import rearrange

"""
Notes/pseudocode about writing attentions with einops in attention.py. 
I only implement decoder-only and encoder-only architectures, 
since they are used in GPT and ViT. Code heavily inspired by karpathy/mingpt
Maybe later ill make a full transformer for translation and stuff.
"""


class Config():
    # model parameters
    m = None # width of input sequence
    k = 64 # key dimension size
    v = 64 # value dimension size
    d = 512 # dimension of hidden state between blocks
    h = 8 # number of heads
    b = 1 # batch dimension
    d_ff = 1024 # size of fully-connected layer
    n_encoders = 6 # number of encoder layers
    n_decoders = 6 # number of decoder layers
    max_seq_length = 1024 # max m

    # hyperparameters for training
    lr = 1e-6 # use lr scheduler later
    dropout = 0.1
    adam_params = {
        'beta_1': 0.9,
        'beta_2': 0.98,
        'epsilon': 10e-9
    }

    # other parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 30000 # shared vocab, if doing NLP
    is_decoder = False

    def init(self, **kwargs): 
        self.__dict__.update(kwargs)


class FeedForward(Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = torch.nn.Linear(config.d, config.d_ff)
        self.fc2 = torch.nn.Linear(config.d_ff, config.d)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) 
        return self.fc2(x)


class AddNorm(Module):
    def __init__(self, config):
        super().__init__()
        self.norm = torch.nn.LayerNorm(config.d)

    def forward(self, x_prev, x):
        x = x_prev + x
        return self.norm(x)


class MHA(torch.nn.Module):
    
    """ 
    Multi Head Attention for the Encoder. 

    Params:
    m: length of the input to the Encoder.
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

    def __init__(self, config, mask=False):
        super().__init__()
        self.P_q = torch.nn.init.kaiming_uniform_(torch.empty(config.h, config.d, config.k))
        self.P_k = torch.nn.init.kaiming_uniform_(torch.empty(config.h, config.d, config.k))
        self.P_v = torch.nn.init.kaiming_uniform_(torch.empty(config.h, config.d, config.v))
        self.P_o = torch.nn.init.kaiming_uniform_(torch.empty(config.h, config.d, config.v))
        self.register_buffer('mask', torch.tril(torch.ones((config.d, config.d)))).view(1, 1, config.d, config.d)

    def forward(x):
        # linear layers before attention
        Q = torch.einsum('bmd, hdk -> bhmk', x, P_q)
        K = torch.einsum('bmd, hdk -> bhmk', x, P_k)
        V = torch.einsum('bmd, hdv -> bhmv', x, P_v)
        # attention
        logits = torch.einsum('bhmk, bhmk -> bhmm', Q, K)
        logits = logits / torch.sqrt(config.k)
        logits = self.mask(logits)
        key_weights = torch.nn.Softmax(logits, dim=-1)
        o = torch.einsum('bhmm, bhmv -> bhmv', key_weights, V)
        # linear layer after attention
        y = torch.einsum('bmhv, hdv -> bmd', o, P_o)
        return y

    @staticmethod
    def mask(self, x):
        seq_length = x.shape[-1]
        return x.masked_fill(self.mask[:, :, :seq_length, :seq_length], float('-inf') )


class Block(Module):
    """
    A transformer block. Can act as both an encoder (ViT) or standalone decoder (GPT).
    As far as i can tell, the only major difference between an encoder and the "decoder" used in GPT
    is the mask. To make this a decoder, set mask=True
    """

    def __init__(self, config, mask=False):
        super().__init__()
        self.mha = MHA(config, mask=mask)
        self.add_norm = AddNorm(config)
        self.ff = FeedForward(config)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x_prev = x
        if self.dropout:
            x = self.dropout(x) # dropout after attention
        x = self.add_norm(x_prev, x)
        x_prev = x
        x = self.ff(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.add_norm(x_prev, x)
        return x


class Transformer(Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embedding = torch.nn.Embedding(config.vocab_size, config.d, padding_idx = 3)
        self.pos_encoding = PositionalEncoder(config)
        self.encoder = nn.Sequential(
            *[Block(config, mask=True) for i in range(config.n_encoders)])
        self.device = config.device
        self.head = nn.Sequential(
            FeedForward(config),
            torch.nn.Linear(config.d, config.vocab_size)
        )  
        self.dropout = config.p_drop
            
    def forward(self, _input):
        # use learned positional encodings
        pos = torch.arange(0, d, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        pos_emb = self.pos_encoding(pos)
        _input = self.input_embedding(_input)
        _input = self.dropout(_input + pos_emb)
        encoder_output = self.encoder(_input)
        # The head can be modified for different purposes.
        # for example, VIT can use a classifier head instead. 
        _output = self.head(encoder_output)
        return _output

class ViT(Module):
    def __init__(self, config):
        super().__init__()
        # TODO add image tokenizer. Maybe even move the embedding stuff out of Transformer class
        self.forward = None
        self.dropout = config.p_drop
            
    def forward(self, _input):
        # use learned positional encodings
        pos = torch.arange(0, d, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        pos_emb = self.pos_encoding(pos)
        _input = self.input_embedding(_input)
        _input = self.dropout(_input + pos_emb)
        encoder_output = self.encoder(_input)
        # The head can be modified for different purposes.
        # for example, VIT can use a classifier head instead. 
        _output = self.head(encoder_output)
        return _output