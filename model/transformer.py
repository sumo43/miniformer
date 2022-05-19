from torch.nn import Module
import torch
import numpy as np
import torch.nn as nn
import cv2
import math
import time

from model.attention import MultiHeadAttention


class EncoderBlock(Module):

    def __init__(self, n_d=512, n_heads=8):

        pass


class TransformerEncoder(Module):
    def __init__(self, n_d=512, n_heads=8, n_l=6):

        self.encoder = nn.Sequential(
            [EncoderBlock(n_d, n_heads) for i in range(n_l)])

        pass

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
