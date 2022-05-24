from torch import nn
import torch

class PositionalEncoder(nn.Module):
    def __init__(self, mp):
        super().__init__()

        self.d_model = mp.d_model
    
    def forward(self, x):

        assert x.shape[1] == self.d_model

        PE = torch.zeros(x.shape, dtype=torch.float32)

        return x

# these parameters are all you need. we pass this class to various functions
class ModelParams():
    def __init__(self, d_model=512, h=8, n_encoders=6, n_decoders=6):
        self.d_model = d_model
        self.h = h
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders

        # d_k = d_v = d_model / h
        self.d_v = self.d_k = d_model // h

        self.d_ff = 0
        self.train_steps = 0
        self.dropout = 0.1
        self.adam_params = {
            'beta_1': 0.9,
            'beta_2': 0.98,
            'epsilon': 10e-9
        }

    def set_ds_size(self, ds_size):
        self.ds_size = ds_size
    
    def set_eng_vocab_size(self, size):
        self.eng_vocab_size = size
    
    def set_sp_vocab_size(self, size):
        self.sp_vocab_size = size
    
