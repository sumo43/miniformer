# Miniformer

Minimal Transformer re-implementation inspired by minGPT. Can be used as a Language Model or a ViT image classifier.

## Description

I made this repository to help me learn about transformers and attention, so I might end up adding other implementation of transformer architectures later on. For now, I only implemented decoder-only and encoder-only architectures, which are used in GPT and ViT respectfully. Code inspired by karpathy/mingpt.

## Setup

```
python3 setup.py develop
```

### Examples

#### ```chargpt```
  * Shakespeare text generation with decoder-only transformer. Benchmark taken from ```karpathy/mingpt```
  
#### ```ViT```
  * Vanilla ViT implementation for MNIST.

### Papers Used
* [Transformer Paper](https://arxiv.org/abs/1706.03762)
* [Fast Transformer Decoding Paper](https://arxiv.org/abs/1911.02150)
* [ViT Paper](https://arxiv.org/pdf/2010.11929.pdf)

### Links
* [karpathy/minGPT](https://github.com/karpathy/mingpt)
* [PyTorch transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
