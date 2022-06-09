import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from utils.data import load_data
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
import os


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
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOK_FILE_LOC)
            
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
            
        raw_en_ds = load_data(EN_DEV_LOC)
        raw_es_ds = load_data(ES_DEV_LOC)

        mp.en_vocab_size = len(raw_en_ds)
        mp.es_vocab_size = len(raw_es_ds)

        print(raw_en_ds[0])
        print(raw_en_ds[0])

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        train_en_ds = torch.utils.data.DataLoader(raw_en_ds, batch_size=32, shuffle=True, collate_fn = lambda x: tokenizer(x, padding='longest', return_tensors='pt'))
        train_es_ds = torch.utils.data.DataLoader(raw_en_ds, batch_size=32, shuffle=True, collate_fn = lambda x: tokenizer(x, padding='longest', return_tensors='pt')) 


        print

        self.train_en_ds = train_en_ds
        self.train_es_ds = train_es_ds

        """
        self.train_en_ds_iter = iter(self.train_en_ds)
        self.train_es_ds_iter = iter(self.train_es_ds)
        """

        self.tokenizer = tokenizer

    def get_en(self):
        return self.train_en_ds

    def get_es(self):
        return self.train_es_ds

    def get_tokenizer(self):
        return self.tokenizer