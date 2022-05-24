import torchtext
import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
import io

def preprocess_data(mp, train, val, test=False):
    if test:
        train = train[:100]
        val = val[:100]

    en_tokenizer = get_tokenizer('moses', language='en')    
    train_iter = iter(train)

    en_vocab = build_vocab_from_iterator(map(en_tokenizer, train_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    # these will be our inputs. list of numbers corr. to list of numbers in spanish vocab
    in_ds = []

    for i in train:
        e = en_tokenizer(i)
        e = ['<bos>', *e, '<eos>']
        in_ds.append(torch.tensor([en_vocab[a] for a in e]))

    es_tokenizer = get_tokenizer('moses', language='es')
    val_iter = iter(val)
    es_vocab = build_vocab_from_iterator(map(es_tokenizer, val_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    out_ds = []
    
    for i in val:
        e = es_tokenizer(i)
        e = ['<bos>', *e, '<eos>']
        out_ds.append(torch.tensor([es_vocab[a] for a in e]))

    mp.set_en_vocab_size(len(en_vocab))
    mp.set_es_vocab_size(len(es_vocab))

    return in_ds, out_ds, en_vocab, es_vocab

def postprocess_data(mp, data, vocab):

    data = data.argmax(dim=1)

    itos = vocab.get_itos()

    return [itos[i] for i in data]