from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import time
from einops.einops import rearrange
import einops

"""
Notes/pseudocode about writing attentions with einops in notes.py. 
I only implement decoder-only and encoder-only architectures, 
since they are used in GPT and ViT. Code inspired by karpathy/mingpt
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

    def __init__(self, **kwargs): 
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
        self.config = config
        self.P_q = torch.nn.init.normal_(torch.empty(config.h, config.d, config.k, requires_grad=True))
        self.P_k = torch.nn.init.normal_(torch.empty(config.h, config.d, config.k, requires_grad=True))
        self.P_v = torch.nn.init.normal_(torch.empty(config.h, config.d, config.v, requires_grad=True))
        self.P_o = torch.nn.init.normal_(torch.empty(config.h, config.d, config.v, requires_grad=True))
        self.register_buffer('triu_mask', torch.tril(torch.ones((config.d, config.d))).view(1, 1, config.d, config.d))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # linear layers before attention
        Q = torch.einsum('bmd, hdk -> bhmk', x, self.P_q)
        K = torch.einsum('bmd, hdk -> bhmk', x, self.P_k)
        V = torch.einsum('bmd, hdv -> bhmv', x, self.P_v)
        # attention
        logits = torch.einsum('bhmk, bhnk -> bhmn', Q, K)
        logits = logits / torch.sqrt(torch.tensor(self.config.k))
        logits = self.mask(logits)
        key_weights = self.softmax(logits)
        o = torch.einsum('bhmm, bhmv -> bhmv', key_weights, V)
        # linear layer after attention
        y = torch.einsum('bhmv, hdv -> bmd', o, self.P_o)
        return y

    def mask(self, x):
        seq_length = x.shape[-1]
        return x.masked_fill(self.triu_mask[:, :, :seq_length, :seq_length], float('-inf') )


class Block(Module):
    """
    A transformer block. Can act as both an encoder (ViT) or standalone decoder (GPT).
    As far as i can tell, the only major difference between an encoder and the "decoder" used in GPT
    is the mask. To make this a decoder, set mask=True
    """

    def __init__(self, config, mask=False):
        super().__init__()
        self.mha = MHA(config, mask=mask)
        self.add_norm_1 = AddNorm(config)
        self.add_norm_2 = AddNorm(config)
        self.ff = FeedForward(config)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        x_prev = x
        x = self.mha(x)
        if self.dropout:
            x = self.dropout(x) # dropout after attention
        x = self.add_norm_1(x_prev, x)
        x_prev = x
        x = self.ff(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.add_norm_2(x_prev, x)
        return x


class Transformer(Module):
    def __init__(self, config):
        super().__init__()
        self.input_embedding = torch.nn.Embedding(config.vocab_size, config.d)
        self.pos_embedding = torch.nn.Embedding(config.max_seq_length, config.d)
        self.config = config
        self.encoder = nn.Sequential(
            *[Block(config, mask=True) for i in range(config.n_decoders)])
        self.device = config.device
        self.head = nn.Sequential(
            torch.nn.Linear(config.d, config.d * 4),
            torch.nn.Linear(config.d * 4, config.vocab_size)
        )  
        self.dropout = torch.nn.Dropout(config.dropout)
        self.ln = torch.nn.LayerNorm(config.d)
         
    def forward(self, _input):
        pos = torch.arange(0, _input.shape[-1], dtype=torch.long, device=self.config.device).unsqueeze(0) # shape (1, t)
        # use learned positional encodings
        seq_length = _input.shape[-1]
        #pos_emb = self.pos_encoding(self.pos)
        input_embeddings = self.input_embedding(_input)
        pos_embeddings = self.pos_embedding(pos)
        _input = self.dropout(input_embeddings + pos_embeddings)
        encoder_output = self.encoder(_input)
        encoder_output = self.ln(encoder_output)
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
        print(_input.shape)
        encoder_output = self.encoder(_input)
        # The head can be modified for different purposes.
        # for example, VIT can use a classifier head instead. 
        _output = self.head(encoder_output)
        return _output