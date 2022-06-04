import torchtext
import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.utils import download_from_url, extract_archive
from utils.general import to_words
from utils.data import load_data
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset, DataLoader
import io
import os

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

EN_DEV_LOC = os.path.join('data', 'opus.en-es-dev.en')
ES_DEV_LOC = os.path.join('data', 'opus.en-es-dev.es')
EN_TRAIN_LOC = os.path.join('data', 'opus.en-es-train.en')
ES_TRAIN_LOC = os.path.join('data', 'opus.en-es-train.es')
EN_TEST_LOC = os.path.join('data', 'opus.en-es-test.es')
ES_TEST_LOC = os.path.join('data', 'opus.en-es-test.es')
TOK_FILE_LOC = os.path.join('data', 'tokenizer-wiki.json')

# this just includes both datasets as torch.nn.Dataset 
class TransformerDataset:

    def __init__(self, mp):

        self.en_tokenizer = None
        self.es_tokenizer = None

        self.en_data = None
        self.es_data = None

        self.dev = None

        if os.path.exists(TOK_FILE_LOC):
            tokenizer = Tokenizer.from_file(TOK_FILE_LOC)
        else:
            tokenizer = Tokenizer(BPE(unk_token="[UNK]", bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"])

            files = [
                EN_DEV_LOC,            
                ES_DEV_LOC, 
                EN_TRAIN_LOC,
                ES_TRAIN_LOC, 
                EN_TEST_LOC,
                ES_TEST_LOC, 
            ]

            tokenizer.post_processor = TemplateProcessing(
                single="[BOS] $A [EOS]",
                special_tokens=[
                ("[UNK]", 0),
                ("[BOS]", 1),
                ("[EOS]", 2),
                ("[PAD]", 3),
            ],
            )

            tokenizer.train(files, trainer)
            tokenizer.save(TOK_FILE_LOC)
            tokenizer.enable_padding(pad_id=3, pad_token='[PAD]')
        
        raw_en_ds = load_data(EN_DEV_LOC)
        raw_es_ds = load_data(ES_DEV_LOC)

        train_en_ds = torch.utils.data.DataLoader(raw_en_ds, batch_size=32, shuffle=True, collate_fn = lambda x: tokenizer.encode_batch(x))
        train_es_ds = torch.utils.data.DataLoader(raw_en_ds, batch_size=32, shuffle=True, collate_fn = lambda x: tokenizer.encode_batch(x)) 

        for item in train_en_ds:

            item = [torch.tensor(a.ids) for a in item]

            print(item)

            print(x)
            print(y)

            print(x.tokens)
            print(y.tokens)

            return

        print('tokenizer loaded')
        print(tokenizer.encode('Name jeff'))



def get_vocabs(mp, _input, _output, test=True):
    if test:
        _input = _input[-10000:]
        _output = _output[-10000:]

    en_tokenizer = get_tokenizer('moses', language='en')    
    train_iter = iter(_input)

    en_vocab = build_vocab_from_iterator(map(en_tokenizer, train_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    es_tokenizer = get_tokenizer('moses', language='es')
    val_iter = iter(_output)
    es_vocab = build_vocab_from_iterator(map(es_tokenizer, val_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    return en_vocab, es_vocab



def preprocess_data(mp, train, val, test=False):

    if test:
        train = train[-10000:]
        val = val[-10000:]

    en_tokenizer = get_tokenizer('moses', language='en')    
    train_iter = iter(train)
    batch_size = mp.batch_size
    device = mp.device

    en_vocab = build_vocab_from_iterator(map(en_tokenizer, train_iter), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    # these will be our inputs. list of numbers corr. to list of numbers in spanish vocab

    in_ds = []

    curr_batch = []

    for it, i in enumerate(train):

        e = en_tokenizer(i)
        e = ['<bos>', *e, '<eos>']
        num_pad_tokens = mp.max_seq_length - len(e)
        [e.append('<pad>') for i in range(num_pad_tokens)] 
        curr_batch.append(torch.tensor([en_vocab[a] for a in e]).unsqueeze(0).to(device))

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
        curr_batch.append(torch.tensor([es_vocab[a] for a in e]).unsqueeze(0).to(device))

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