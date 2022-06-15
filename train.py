from utils.general import ModelParams
from utils.data import load_data
from utils.preproc import TransformerDataset
from model.transformer import Transformer
from model.trainer import TransformerTrainer
import torch

"""

Train Config
The big transformer model in the paper takes about 3.5 days on 8 P100 GPUs (28 days P100)
The smaller, regular model, which has everything reduced by 2x, takes about 7.5x less flops to train, so about 4 days with a P100
extrapolating this to a smaller model with d_model = 128, 4 heads, I'm hoping for another 8-16x reduced time to train
I have access to a T4, which is about 80% the performance of a K100, 
so I'm hoping to train the model to be an okay-ish english to spanish translator in about a day

"""

d_model = 128
h= 4
n_encoders=4
n_decoders=4
d_ff= 256
# max length for both input and output, including <s> </s> 
# we pad it up to this length
max_seq_length = 4096
batch_size = 32
NUM_BATCHES = 20

mp = ModelParams(d_model=d_model, 
    h=h, 
    n_encoders=n_encoders, 
    n_decoders=n_decoders, 
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    batch_size=batch_size
    )

mp.device = 'cuda'

td = TransformerDataset(mp)
device = mp.device
t = Transformer(mp).to(device)
tr = TransformerTrainer(mp, t, td)

tr.train()
torch.save(t, 'model.pt')
files.download('model.pt')