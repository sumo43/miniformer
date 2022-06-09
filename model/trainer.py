from torch import nn
import torch
import math
from time import sleep
from random import shuffle
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from transformers.utils.dummy_tf_objects import TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from model.transformer import Transformer
from utils.preproc import TransformerDataset
from utils.general import ModelParams

class TransformerTrainer:
    def __init__(self, mp: ModelParams, model: Transformer, td: TransformerDataset):
        
        # other params 

        self.mp = mp

        # hyperparams 

        self.beta_1 = mp.adam_params['beta_1']
        self.beta_2 = mp.adam_params['beta_2']
        self.epsilon = mp.adam_params['epsilon']
        self.dropout = mp.dropout
        self.train_steps = mp.train_steps or 5000
        self.d_model = mp.d_model

        # model

        self.model = model

        # optimizer and loss

        self.optimizer = torch.optim.Adam(model.parameters(), betas=(self.beta_1, self.beta_2))
        self.loss = torch.nn.CrossEntropyLoss()

        # dataset and tokenizer

        self.td = td
        self.ds = self.td.get_ds()
        self.tokenizer = self.td.get_tokenizer()

    def test_train(self):

        for (train_example, label_example) in self.ds:

            x = train_example['input_ids']
            y = label_example['input_ids']

            print(x.shape)
            print(y.shape)
            
            y_pred = self.model(x, y)

            return
    
    def train(self):

        for i in tqdm(range(len(self.in_ds))):

            ind = i % len(self.in_ds)

            if i % 50 == 0:
                _print = True
            else:
                _print = False
            
            try:
                self.train_iteration(self.model, self.in_ds[ind], self.out_ds[ind], self.loss, self.optimizer, _print=_print, i=i)

            except Exception as e:
                print(e)
                continue
        
        
    def train_iteration(self, model, x, y, loss_fn, optimizer, _print=False, i=0):

        # feed the outputs shifted right, but we later compute loss w/ non-shifted
        y_pred = model(x, y)

        #print(x.shape)

        if _print:
            y_old = y
            y_pred_old = y_pred
        
        y = y[:, 1:]
        y_pred = y_pred[:, :-1]
        """
        if i > 0:
            lr = math.pow(float(self.d_model), -0.5) * math.pow(float(i), -0.5)

            for g in optimizer.param_groups:
                g['lr'] = lr
        """
        optimizer.zero_grad()

        batch_size, seq_len, vocab_size = y_pred.shape

        y_pred = y_pred.ravel().view(-1, vocab_size)
        y = y.ravel().view(-1)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


        #print(y_old.shape)
        #print(y_pred_old.shape)

        if _print:
            print(to_words(x[0], self.eng_vocab))
            print(to_words(y_old[0], self.sp_vocab))
            y_pred_arg = torch.argmax(y_pred_old, dim=2)
            print(to_words(y_pred_arg[0], self.sp_vocab))
            

        if _print:
            print(loss.item())

def to_tokens(x, eng_vocab):

    _out = []
    for word in x:
        _out.append(eng_vocab(word))
    
    return torch.tensor(_out)

class TransformerEvaluator:

    def __init__(self, mp, model, data):

        # greedy search
        self.mp = mp
        self.model = model
        self.eng_vocab, self.sp_vocab = data

        self.eng_vocab.set_default_index(0)
        self.sp_vocab.set_default_index(0)
    
    def eval(self, x):

        # eval until </s> token or reach max seq length
        en_tokenizer = get_tokenizer('moses', language='en')
        x = en_tokenizer(x)

        [x.append('<pad>') for i in range(128 - len(x))]

        x_in = torch.tensor(self.eng_vocab(x))

        y = []

        y.append('<bos>')


        #[y.append('<pad>') for i in range(127)]

        x_in = x_in.unsqueeze(0)
        y_in = torch.tensor(self.sp_vocab(y)).unsqueeze(0)

        for i in range(127):
            val, ind = torch.topk(self.model(x_in, y_in), 5, dim=2)

            print(ind)

            print(ind.shape)
            for j in range(5):
                print(to_words(ind[:, :, j], self.sp_vocab))

            print(to_words(x[0], self.eng_vocab))
            print(to_words(y_in[0], self.sp_vocab))

            return

        print(y_in)

    def eval_iteration(self, model, x, y, loss_fn, optimizer):

        y_pred = model(x, y)
        optimizer.zero_grad()

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        print(loss.item())