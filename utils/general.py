from torch import nn
import torch
import math

from time import sleep

VERY_SMOL_NUM = -33209582095375352525228572587289578295.

class PositionalEncoder(nn.Module):
    def __init__(self, mp):
        super().__init__()

        self.d_model = mp.d_model
        self.max_seq_length = mp.max_seq_length

        # we make the max sequence length 128
        self.encoding_tensor = torch.zeros((self.max_seq_length, self.d_model), requires_grad=False)
        for pos in range(self.encoding_tensor.shape[0]):
            for i in range(self.encoding_tensor.shape[1]):
                if i % 2 == 0:
                    self.encoding_tensor[pos, i] = math.sin(pos / (math.pow(10000, ((2 * i / self.d_model)))))
                else:
                    self.encoding_tensor[pos, i] = math.cos(pos / (math.pow(10000, ((2 * i / self.d_model)))))
        
        self.encoding_tensor = self.encoding_tensor.unsqueeze(0)
    
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
    def __init__(self, mp, model, data):

        self.mp = mp

        self.beta_1 = mp.adam_params['beta_1']
        self.beta_2 = mp.adam_params['beta_2']
        self.epsilon = mp.adam_params['epsilon']

        self.dropout = mp.dropout
        self.train_steps = mp.train_steps or 5000

        self.model = model
        self.in_ds, self.out_ds, self.eng_vocab, self.sp_vocab = data

        self.optimizer = torch.optim.Adam(model.parameters(), betas=(self.beta_1, self.beta_2))
        self.loss = torch.nn.CrossEntropyLoss()

        self.es_vocab_size = mp.es_vocab_size

    def train(self):
        for i in range(len(self.train_ds)):

            ind = i % len(self.in_ds)

            if i % 100 == 0:
                _print = True
            else:
                _print = False
            self.train_iteration(self.model, self.in_ds[ind], self.out_ds[ind], self.loss, self.optimizer, _print=_print)

    def train_iteration(self, model, x, y, loss_fn, optimizer, _print=False):
        # shift right 

        y = y[:, :-1]

        y_pred = model(x, y)

        if _print:
            y_old = y
            y_pred_old = y_pred

        y_pred = y_pred[:, 1:, :]

        optimizer.zero_grad()

        batch_size, seq_len, vocab_size = y_pred.shape

        y_pred = y_pred.ravel().view(-1, vocab_size)
        y = y.ravel().view(-1)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if _print:
            y_pred_arg = torch.argmax(y_pred_old, dim=2)
            print(f'x: {x[0]} y: {y_old[0]} y_pred: {y_pred_arg[0]}')

        if _print:
            print(loss.item())

class TransformerEvaluator:

    def __init__(self, mp, model, data):

        # greedy search
        self.mp = mp
        self.model = model
        _, _, self.eng_vocab, self.sp_vocab = data
    
    def eval(self):

        # eval until </s> token or reach max seq length

        for i in range(1, 10000): 
            self.eval_iteration(self.model, self.in_ds[ind], self.out_ds[ind], self.loss, self.optimizer)

    def eval_iteration(self, model, x, y, loss_fn, optimizer):

        y_pred = model(x, y)
        optimizer.zero_grad()

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        print(loss.item())


    
