from torch import nn
import torch
import math

class PositionalEncoder(nn.Module):
    def __init__(self, mp):
        super().__init__()

        self.d_model = mp.d_model

        # we make the max sequence length 128
        self.encoding_tensor = torch.zeros((128, 512), requires_grad=False)
        for pos in range(self.encoding_tensor.shape[0]):
            for i in range(self.encoding_tensor.shape[1]):
                if i % 2 == 0:
                    self.encoding_tensor[pos, i] = math.sin(pos / (math.pow(10000, ((2 * i / self.d_model)))))
                else:
                    self.encoding_tensor[pos, i] = math.cos(pos / (math.pow(10000, ((2 * i / self.d_model)))))
    
    def forward(self, x):
        assert x.shape[1] == self.d_model

        x = x + self.encoding_tensor[:x.shape[0], :]

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
    
    def set_en_vocab_size(self, size):
        self.en_vocab_size = size
    
    def set_es_vocab_size(self, size):
        self.es_vocab_size = size

class TransformerTrainer:
    def __init__(self, mp, model):

        self.beta_1 = mp.adam_params.beta_1
        self.beta_2 = mp.adam_params.beta_2
        self.epsilon = mp.adam_params.epsilon

        self.dropout = mp.dropout
        self.train_steps = mp.train_steps

        optimizer = torch.nn.Adam(model.parameters(), betas=(self.beta_1, self.beta_2))
        loss = torch.nn.CrossEntropyLoss()

        for i in range(self.train_steps):
            self.train_iteration(model, x, y, loss, optimizer)
        
        return 

    def train_iteration(self, model, x, y, loss_fn, optimizer):
        y_pred = model(x)
        optimizer.zero_grad()
        loss = loss_fn(y, y_pred)
        loss.backward()
        optimizer.step()
        return


    
