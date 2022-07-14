from utils.general import ModelParams, TransformerTrainer, TransformerEvaluator
from utils.data import load_data
from utils.preproc import preprocess_data, postprocess_data, to_string, get_vocabs, TransformerDataset
from model.transformer import Transformer
import torch



"""
ViT with classifier head to classify MNIST.
"""

vit_config = {
    # model parameters
    'm' : None,
    'k' : 32, # key dimension size
    'v' : 32, # value dimension size
    'd' : 128, # dimension of hidden state between blocks
    'h' : 4, # number of heads
    'd_ff' : 128, # size of fully-connected layer
    'n_encoders' : 2, # number of encoder layers
    'n_decoders' : 2, # number of decoder layers
    'max_seq_length' : 1024, # max m
    # hyperparameters for training
    'lr' : 1e-5, # use lr scheduler later
    'vocab_size': 100
}

d_model = 128
h= 4
n_encoders=4
n_decoders=4
d_ff= 256
# max length for both input and output, including <s> </s> 
# we pad it up to this length
max_seq_length = 128
batch_size = 32
NUM_BATCHES = 10

mp = ModelParams(d_model=d_model, 
    h=h, 
    n_encoders=n_encoders, 
    n_decoders=n_decoders, 
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    batch_size=batch_size
    )


td = TransformerDataset(mp)

"""
inputs, outputs = load_data(mp)

print('loaded date')

eng_vocab, sp_vocab = get_vocabs(mp, inputs, outputs)

print('loaded vocabs')
#in_ds, out_ds, eng_vocab, sp_vocab = preprocess_data(mp, inputs, outputs, test=True)

t = torch.load('weights/weights.pt', map_location=torch.device(mp.device))
    
ev = TransformerEvaluator(mp, t, (eng_vocab, sp_vocab))

ev.eval("I will drive my car.")
"""