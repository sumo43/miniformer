#### Transformer PROS
- scales well. https://arxiv.org/pdf/2205.06175.pdf
- You can use it for a lot of stuff. Translation, vision transformer, latent space representations for other decoders
- faster than RNNs on sequence data, since you can feed the entire input sequence at once

#### Transformer CONS
- Need to be big to be good? We investigate
- RNNs have better parameter efficiency

#### Model Format and details

D = 512, dimension of dense
N = 6, Number of encoder and decoder blocks
h = 8, amt of attn heads






