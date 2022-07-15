import torch
import os
from collections import Counter
from miniformer.model import Transformer, Config
from miniformer.trainer import Trainer

"""
GPT with character-level embeddings to generate shakespeare.
"""

DATA_FILE = os.path.join('..', 'data', 'input.txt')
SPECIAL_TOKEN = 0 # can use this for EOS, padding, etc...

chargpt_config = {
    # model parameters. This may be the smallest viable language model possible.
    'm' : None,
    'k' : 32, # key dimension size
    'v' : 32, # value dimension size
    'd' : 128, # dimension of hidden state between blocks
    'h' : 4, # number of heads
    'd_ff' : 128, # size of fully-connected layer
    'n_encoders' : 4, # number of encoder layers
    'n_decoders' : 4, # number of decoder layers
    'max_seq_length' : 128, # max m
    # hyperparameters for training
    'lr' : 4e-4, # use lr scheduler later
    'vocab_size': 39,
    'epochs': 10,
    'batch_size': 32
}

class CharDataset:
    def __init__(self, path, config):
        self.config = config
        self.data = open(path, 'r').read()
        self.data = self.data.lower() # lowecase only
        # count the occurrences of each char and sort them
        chars = sorted(Counter(list(self.data)).items(), key=lambda a: a[1], reverse=True)
        self.ctoi = {c[0]: i for i,c in enumerate(chars)}
        #self.ctoi['<S>'] = 0 # make space for SPECIAL_TOKEN at 0
        self.config.vocab_size = len(self.ctoi)
        self.itoc = {v:k for k, v in self.ctoi.items()}

    def get_ctoi(self, x):
        return self.ctoi[x]
    
    def get_itoc(self, x):
        return self.itoc[x]
    
    def __getitem__(self, idx):
        data = torch.Tensor(list(map(self.get_ctoi, self.data[idx:idx+config.max_seq_length+1])))\
        .type(torch.LongTensor)\
        .to(self.config.device)
        x = data[:-1]
        y = data[1:]
        return x, y
    
    def __len__(self):
        return len(self.data) - (config.max_seq_length + 1)

config = Config(**chargpt_config)
dataset = CharDataset(DATA_FILE, config)
model = Transformer(config).to(config.device)
trainer = Trainer(dataset, config)
trainer.train(model, num_epochs=10, batch_size=config.batch_size)

# Test the model

ex_text = 'it is not'
tok_text = list(map(dataset.get_ctoi, ex_text))
out_tok = model(torch.tensor(tok_text).type(torch.LongTensor).to(config.device))

print(str(map(dataset.get_itoc, out_tok)))
