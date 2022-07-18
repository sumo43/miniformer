from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import time
from einops.einops import rearrange
import torch.nn.functional as F
import einops

"""
Notes/pseudocode about writing attentions with einops in notes.py. 
I only implement decoder-only and encoder-only architectures, 
since they are used in GPT and ViT. Code inspired by karpathy/mingpt
Maybe later ill make a full transformer for translation and stuff.
"""

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Config():
    # model parameters
    m = None # width of input sequence
    k = 32 # key dimension size
    v = 32 # value dimension size
    d = 192 # dimension of hidden state between blocks
    h = 6 # number of heads
    b = 1 # batch dimension
    d_ff = 128 # size of fully-connected layer
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

class MHA(nn.Module):

    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    Protip: Don't use einops... hard to debug
    """

    def __init__(self, config, mask):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.l1 = nn.Linear(config.d, 3 * config.d)
        self.l2 = nn.Linear(config.d, config.d)
        # regularization
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
                                     .view(1, 1, config.max_seq_length, config.max_seq_length))
        self.d = config.d
        self.h = config.h
        self.config = config

    def forward(self, x):
        config = self.config
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.l1(x).split(self.d, dim=2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        q = q.view(config.batch_size, T, config.h, config.k).transpose(2, 1)
        k = k.view(config.batch_size, T, config.h, config.k).transpose(2, 1)
        v = v.view(config.batch_size, T, config.h, config.v).transpose(2, 1)
        logits = torch.einsum('bhmk, bhnk -> bhmn', q, k)
        logits = logits / torch.sqrt(torch.tensor(self.config.k))
        att = logits.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.einsum('bhmm, bhmv -> bhmv', att, v)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.l2(y))
        return y

class Block(Module):
    """
    A transformer block. Can act as both an encoder (ViT) or standalone decoder (GPT).
    As far as i can tell, the only major difference between an encoder and the "decoder" used in GPT
    is the mask. To make this a decoder, set mask=True
    """

    def __init__(self, config, mask=False):
        super().__init__()
        self.mhastack = torch.nn.Sequential(
            torch.nn.LayerNorm(config.d),
            MHA(config, mask=mask)
        )
        self.linstack = torch.nn.Sequential(
            torch.nn.LayerNorm(config.d),
            torch.nn.Linear(config.d, config.d * 4),
            NewGELU(),
            torch.nn.Linear(config.d * 4, config.d),
            torch.nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.mhastack(x)
        x = x + self.linstack(x)
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
            torch.nn.Linear(config.d, config.vocab_size)
        )  
        self.dropout = torch.nn.Dropout(config.dropout)
        self.ln = torch.nn.LayerNorm(config.d)
         
    def forward(self, _input):
        pos = torch.arange(0, _input.shape[-1], dtype=torch.long, device=self.config.device).unsqueeze(0) # shape (1, t)
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