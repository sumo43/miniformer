from torch import nn
import torch
import math
from time import sleep
from random import shuffle
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer

VERY_SMOL_NUM = -33209582095375352525228572587289578295.

def to_words(_input: torch.Tensor, vocab):

    print(_input)
    itos = vocab.get_itos()
    ret = []

    return [itos[i] for i in _input]

class PositionalEncoder(nn.Module):
    def __init__(self, mp):
        super().__init__()

        self.d_model = mp.d_model
        self.max_seq_length = mp.max_seq_length
        self.device = mp.device

        # we make the max sequence length 128
        self.encoding_tensor = torch.zeros((self.max_seq_length, self.d_model), requires_grad=False)
        for pos in range(self.encoding_tensor.shape[0]):
            for i in range(self.encoding_tensor.shape[1]):
                if i % 2 == 0:
                    self.encoding_tensor[pos, i] = math.sin(pos / (math.pow(10000, ((2 * i / self.d_model)))))
                else:
                    self.encoding_tensor[pos, i] = math.cos(pos / (math.pow(10000, ((2 * i / self.d_model)))))
        
        self.encoding_tensor = self.encoding_tensor.unsqueeze(0).to(self.device)
    
    def forward(self, x):
        assert x.shape[-1] == self.d_model

        x = x + self.encoding_tensor[:, :x.shape[1], :]

        return x

# these parameters are all you need. we pass this class to various functions
class ModelParams():
    def __init__(self, 
        d_model=512, 
        h=8, 
        n_encoders=6, 
        n_decoders=6, 
        d_ff=2048,
        max_seq_length=32,
        batch_size=32,
        train_steps = 1000
        ):

        self.d_model = d_model
        self.h = h
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders
        self.max_seq_length = max_seq_length
        self.batch_size = 32
        self.d_ff = d_ff
        self.train_steps = train_steps

        # d_k = d_v = d_model / h
        self.d_v = self.d_k = d_model // h
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

        print(self.device)

        self.dropout = 0.1
        self.adam_params = {
            'beta_1': 0.9,
            'beta_2': 0.98,
            'epsilon': 10e-9
        }

    def set_ds_size(self, ds_size):
        self.ds_size = ds_size
    
    def set_en_vocab_size(self, size):
        self.en_vocab_size = size
    
    def set_es_vocab_size(self, size):
        self.es_vocab_size = size