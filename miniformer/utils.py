from torch import nn
import torch
import math
from time import sleep
from random import shuffle
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer

def to_words(_input: torch.Tensor, vocab):

    print(_input)
    itos = vocab.get_itos()
    ret = []

    return [itos[i] for i in _input]


    def forward(self, x):
        assert x.shape[-1] == self.d_model

        x = x + self.encoding_tensor[:, :x.shape[1], :]

        return x
