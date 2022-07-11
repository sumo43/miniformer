#Transformer in Pytorch

Transformer from Attention is All you Need for english-to-spanish machine translation.
Trained on [this dataset](https://www.manythings.org/anki/spa-eng.zip)

My goal with this model was to for it to be reasonably good at translating 1-3 sentences in English to Spanish, so I used a small amount of input and output tokens. In addition, I am using a much smaller version of the transformer than the one described in the paper, the exact params are in train.py. 
