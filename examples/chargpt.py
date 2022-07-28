import torch
import os
from collections import Counter
from miniformer.model import Transformer, Config
from miniformer.trainer import Trainer
import torch.nn as nn

"""
GPT with character-level embeddings to generate shakespeare.
"""

DATA_FILE = os.path.join('..', 'data', 'input.txt')

chargpt_config = {
    # model parameters. smaller than chargpt model from mingpt
    'm' : None,
    'k' : 32, # key dimension size
    'v' : 32, # value dimension size
    'd' : 192, # dimension of hidden state between blocks
    'h' : 6, # number of heads
    'n_encoders' : 6, # number of encoder layers
    'n_decoders' : 6, # number of decoder layers
    'max_seq_length' : 128, # max m
    # hyperparameters for training
    'lr' : 5e-4, # use lr scheduler later
    'epochs': 10,
    'batch_size': 64
}

class CharDataset:
    def __init__(self, path, config):
        self.config = config
        self.data = open(path, 'r').read()
        self.data = self.data
        # count the occurrences of each char and sort them
        chars = sorted(Counter(list(self.data)).items(), key=lambda a: a[1], reverse=True)
        self.ctoi = {c[0]: i for i,c in enumerate(chars)}
        self.config.vocab_size = len(self.ctoi)
        self.itoc = {v:k for k, v in self.ctoi.items()}

    def get_ctoi(self, x):
        return self.ctoi[x]
    
    def get_itoc(self, x):
        return self.itoc[x]
    
    def __getitem__(self, idx):
        data = torch.tensor(list(map(self.get_ctoi, self.data[idx:idx+config.max_seq_length+1])), dtype=torch.long)
        x = data[:-1]
        y = data[1:]
        return x, y
    
    def __len__(self):
        return len(self.data) - (config.max_seq_length + 1)


config = Config(**chargpt_config)
dataset = CharDataset(DATA_FILE, config)
model = Transformer(config)
model = model.to(config.device)
opt = model.get_optimizer()
trainer = Trainer(dataset, config)
trainer.train(model, opt, num_epochs=10, batch_size=config.batch_size)