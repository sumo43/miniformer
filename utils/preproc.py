import torchtext
import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
import io

"""
def pad_sentences(mp, ds, tokenizer, voc):

    out_ds = []

    for i in ds:
        e = tokenizer(i)
        e = ['<bos>', *e, '<eos>']
        num_pad_tokens = mp.max_seq_length - len(e)
        [e.append('<pad>') for i in range(num_pad_tokens)]

        out_ds.append(torch.tensor([es_vocab[a] for a in e]))
    
    return out_ds
"""

def preprocess_data(mp, train, val, test=False):

    if test:
        train = train[:10000]
        val = val[:10000]

    en_tokenizer = get_tokenizer('moses', language='en')    
    train_iter = iter(train)
    batch_size = mp.batch_size

    en_vocab = build_vocab_from_iterator(map(en_tokenizer, train_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    # these will be our inputs. list of numbers corr. to list of numbers in spanish vocab

    in_ds = []

    curr_batch = []

    for it, i in enumerate(train):

        e = en_tokenizer(i)
        e = ['<bos>', *e, '<eos>']
        num_pad_tokens = mp.max_seq_length - len(e)
        [e.append('<pad>') for i in range(num_pad_tokens)] 
        curr_batch.append(torch.tensor([en_vocab[a] for a in e]).unsqueeze(0).to('cuda:0'))

        if it != 0 and it % batch_size == 0:
            curr_batch = torch.cat(curr_batch)
            in_ds.append(curr_batch)
            curr_batch = []

    curr_batch = [] 

    es_tokenizer = get_tokenizer('moses', language='es')
    val_iter = iter(val)
    es_vocab = build_vocab_from_iterator(map(es_tokenizer, val_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    out_ds = []
    
    for it, i in enumerate(val):
        e = es_tokenizer(i)
        e = ['<bos>', *e, '<eos>']
        num_pad_tokens = mp.max_seq_length - len(e)
        [e.append('<pad>') for i in range(num_pad_tokens)]
        curr_batch.append(torch.tensor([es_vocab[a] for a in e]).unsqueeze(0).to('cuda:0'))

        if it != 0 and it % batch_size == 0:
            curr_batch = torch.cat(curr_batch)
            out_ds.append(curr_batch)
            curr_batch = []
    
    mp.set_en_vocab_size(len(en_vocab))
    mp.set_es_vocab_size(len(es_vocab))

    return in_ds, out_ds, en_vocab, es_vocab

def postprocess_data(mp, data):
    data = data.argmax(dim=1)
    return data

def to_string(data, vocab):
    itos = vocab.get_itos()
    st = ''
    for i in data:
        st += itos[i] + ' ' 
    return st
