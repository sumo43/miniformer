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
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # dataset and tokenizer

        self.td = td
        self.ds = self.td.get_ds()
        self.val_ds = self.td.get_val_ds()
        self.tokenizer = self.td.get_tokenizer()

        # todo make this not a constatn

        self.vocab_size = 30000
        self.tokenizer = self.td.tokenizer

        # interval over which to run validation loop (2min)#
        # 30000 iterations is about half an hour

        self.val_interval = 30000 // 32

        self.batch_size = mp.batch_size
    
    def train(self):
        for epoch in range(1):
            self._train()

    def _train(self):

        running_loss = 0

        for i, (train_example, label_example) in enumerate(tqdm(self.ds)):
            # training step

            x = train_example['input_ids']
            y = label_example['input_ids']

            y_shifted = y[:, 1:]
            y = y[:, :-1]

            # feed the outputs shifted right, but we later compute loss w/ non-shifted
            y_pred = self.model(x, y)

            # resize everything for CEloss
            y_pred = y_pred.view(-1, self.vocab_size)
            y_shifted = y_shifted.ravel()
            
            self.optimizer.zero_grad()
            loss = self.loss_fn(y_pred, y_shifted)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # code for setting good learning rate from paper
            """
            if i > 0:
                lr = math.pow(float(self.d_model), -0.5) * math.pow(float(i), -0.5)

                for g in optimizer.param_groups:
                    g['lr'] = lr
            """

            # running loss
            if i % 100 == 999:
                running_loss /= 100
                print(f'batch {i} loss: {running_loss}')
                running_loss = 0

            if i % self.val_interval == 0:
                print('validating...')
                self._validate()

    def _validate(self):
    
        #running_loss = 0

        #val_loss = torch.nn.MSELoss()

        for i, (val_example, label_example) in enumerate(tqdm(self.val_ds)):

            starting_sentence = torch.zeros((1, 1))
            starting_sentence = starting_sentence.type(torch.IntTensor)
            starting_sentence[0, 0] = 1

            y_pred = starting_sentence

            x = val_example['input_ids']
            y = label_example['input_ids']
            y_shifted = y[:, 1:]

            while(y_pred.shape[1] != y.shape[1] - 1):

                new_y_pred = torch.argmax(self.model(x, y_pred), -1)[:, -1:]
                y_pred = torch.cat([y_pred, new_y_pred], dim=1)
        
            # resize everything, also make it floaty for loss
            # i know that this loss is stupid, but im too lazy to make cross entropy loss work
            #y_pred_fl = y_pred.ravel().type(torch.FloatTensor)
            #y_shifted_fl = y_shifted.ravel().type(torch.FloatTensor)

            #loss = val_loss(y_pred_fl, y_shifted_fl)
            #running_loss += loss.item()

            # running loss
            if i % 100 == 0:
                #running_loss /= 100
                #print(f'val_batch {i} val_loss: {running_loss}')
                print(self.tokenizer.batch_decode(x)[0])
                print(self.tokenizer.batch_decode(y_pred)[0])
                #running_loss = 0