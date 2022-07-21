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
  * Shakespeare text generation with decoder-only transformer. dataset taken from ```karpathy/mingpt```
```
*PROMPT*: O God, O God!

*GENERATED TEXT*:

KING RICHAM:
That thou art thus.

KING LEWIS XI:
It is no more shall have teach'd me in
The grace! to my wrongether's death, if you must do that?

KATHARINA:
Twas by you, how should his grands are in the court,
Widle they swear and my life beseech your flowers
Agueu in thy weedd as stand this most wreck obedience
Tear on thee short ofth, the than be fathom.
But yet, my fancing.

Go schaen our breast will they were well seeed.
O trurn this shephething hate!
By the weed full of the measures of a
```
  
#### ```ViT```
  * Vanilla ViT implementation for MNIST.

[<img alt="mnist example" width="100px" src="examples/mnist.png" />](https://www.google.com/)

```
LABEL: 5
PREDICTED: 5
```

### Papers Used
* [Transformer Paper](https://arxiv.org/abs/1706.03762)
* [Fast Transformer Decoding Paper](https://arxiv.org/abs/1911.02150)
* [ViT Paper](https://arxiv.org/pdf/2010.11929.pdf)

### Links
* [karpathy/minGPT](https://github.com/karpathy/mingpt)
* [PyTorch transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
* [lucidrains vit-pytorch](https://github.com/lucidrains/vit-pytorch)
