import torch
from miniformer.model import Config, Transformer
from time import sleep

"""
Tests to make sure shapes and stuff are correct
These probably won't catch architecture bugs that may prevent it from converging
"""

test_config = {
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

# default params
config = Config(**test_config)
t = Transformer(config)


optimizer = torch.optim.SGD(t.parameters(), lr=config.lr)
loss= torch.nn.CrossEntropyLoss()

"""

for i in range(1000):

print(torch.argmax(model(test_x), -1))
"""

# sort_dataset: x is [1, 2, 3, 4, 5], y is [5, 4, 3, 2, 1]
x = []
y = []
for i in range(500):
    i = torch.randint(0, 9, (1, 5)).type(torch.LongTensor)
    o = i * 2


    print(i.shape)
    print(o.shape)
    x.append(i)
    y.append(o)

for i in range(1000):
    for j in range(len(x)):
        x_i = x[j]
        y_i = y[j]
        loss.zero_grad()
        y_pred = t(x_i)

        print(x_i)
        print(torch.argmax(y_pred, -1))

        l = loss(y_pred.reshape(-1, 100), y_i.reshape(-1,))
        print(l.item())
        l.backward()
        optimizer.step()




"""

sdpa = SDPAttention(mp)

# fake qkv

Q = torch.randn((1, 1, mp.d_k))
K = torch.randn((1, 1, mp.d_k))
V = torch.randn((1, 1, mp.d_v))

sdpa(Q, K, V)

mha = MultiHeadAttention(mp)

Q = torch.randn((1, 1, mp.d_model))
K = torch.randn((1, 1, mp.d_model))
V = torch.randn((1, 1, mp.d_model))

mha(Q, K, V)

eb = EncoderBlock(mp)
print(eb(Q).shape)

e = TransformerEncoder(mp)
print(e(Q).shape)

db = DecoderBlock(mp)

inputs = (Q, K)

print(db(inputs)[0][1].shape)

inputs, outputs = load_data(mp)
in_ds, out_ds, eng_vocab, sp_vocab = preprocess_data(mp, inputs, outputs, test=True)

#d = TransformerDecoder(mp)

#print(d(inputs).shape)

#megatron = Transformer(n_dim, n_heads, ...)

dataset = zip(inputs, outputs)

config = ModelConfig()

t = Transformer(mp)
x = t(in_ds[0], out_ds[0])
print(x.shape)

x = postprocess_data(mp, x)
print(to_string(x, sp_vocab))

x = torch.ones((4, 4), dtype=torch.float)

def mask(x):
    tri_mask = torch.triu(torch.ones(size=x.shape, dtype=torch.float)) * VERY_SMOL_NUM
    print(tri_mask)
    x = x + tri_mask
    return x

tr = TransformerTrainer(mp, t)

print('those tests are all we need')
"""
