import torch
import os
from collections import Counter
from miniformer.model import ViT, Config
from miniformer.trainer import ViTTrainer
import torchvision

"""
GPT with character-level embeddings to generate shakespeare.
"""

DATA_FILE = os.path.join('..', 'data', 'input.txt')

vit_config = {
    # model parameters. same as mingpt's gpt-mini config
    'm' : None,
    'k' : 64, # key dimension size
    'v' : 64, # value dimension size
    'd' : 384, # dimension of hidden state between blocks
    'h' : 6, # number of heads
    'n_encoders' : 6, # number of encoder layers
    'n_decoders' : 6, # number of decoder layers
    'max_seq_length' : 128, # max m
    # hyperparameters for training
    'lr' : 4e-4, # use lr scheduler later
    'epochs': 10,
    'batch_size': 64,
    'n_threads': 4,
    'vocab_size': 10
}

config = Config(**vit_config)

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

model = ViT(config)
model = model.to(config.device)
trainer = ViTTrainer(dataset, config)
trainer.train(model, num_epochs=10, batch_size=config.batch_size)