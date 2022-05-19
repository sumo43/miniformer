

#### Problems with Transformers
- Need to be very big to be good
- RNNs have better parameter efficiency

D = 512, dimension of dense
N = 6, Number of encoder and decoder blocks
h = 8, amt of attn heads

#### Encoder

Input Embedding + Positional Encoding (Word2Vec?)

Sub Layer 1:
Multi Head Attention -> Add residual Connection -> Layer Norm

Sub Layer 2:
Feed Forward NN 512->512 -> Layer Norm

#### Attention Mechanism











