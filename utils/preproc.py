import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
import io

"""

en_tokenizer = get_tokenizer('spacy', language='en')
es_tokenizer = get_tokenizer('spacy', language='es')

"""

def preprocess_data(train, val):
    eng_tokenizer = get_tokenizer('moses', language='en')    
    train_iter = iter(train)
    eng_vocab = build_vocab_from_iterator(map(eng_tokenizer, train_iter), specials=['<unk>', '<pad>'])

    es_tokenizer = get_tokenizer('moses', language='es')
    val_iter = iter(val)
    sp_vocab = build_vocab_from_iterator(map(es_tokenizer, val_iter), specials=['<unk>', '<pad>'])

    return eng_vocab, sp_vocab


