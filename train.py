"""

Train Config
The big transformer model in the paper takes about 3.5 days on 8 P100 GPUs (28 days P100)
The smaller, regular model, which has everything reduced by 2x, takes about 7.5x less flops to train, so about 4 days with a P100
extrapolating this to a smaller model with d_model = 256, 4 heads, I'm hoping for another ~8x reduced time to train
I have access to a T4, which is about 80% the performance of a K100, 
so I'm hoping to train the model to be an okay-ish english to spanish translator in 1-2 days

"""
