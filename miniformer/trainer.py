import torch
from tqdm import tqdm

# TODO makek the two trainers subclass from a shared Trainer class for OOP points

class Trainer:
    def __init__(self, dataset, config):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True) # map-style dataset
        self.config = config
        self.dataset = dataset

    def train(self, model, **kwargs):
        params = model.parameters()

        optim_groups = [
            {"params": [pn for pn in params if isinstance(pn, torch.nn.Linear) or isinstance(pn, torch.nn.Parameter)], "weight_decay": 0.1},
            {"params": [pn for pn in params if not isinstance(pn, torch.nn.Linear and not isinstance(pn, torch.nn.Parameter))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=(0.9, 0.98))
        loss_fn = torch.nn.CrossEntropyLoss()
        running_loss = 0
        model.train()

        

        # this works for chargpt, but needs to be generalized for models that need both x, y as inputs, like translators
        for epoch in range(self.config.epochs):
            for i, (x, y) in enumerate(self.dataloader):
                print(i)
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                y_pred = model(x)
                y_pred_argmax = torch.argmax(y_pred, -1)
                model.zero_grad(set_to_none=True)
                loss = loss_fn(y_pred.view(-1, self.config.vocab_size), y.view(-1,))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                running_loss += loss.item()
                print(f'epoch {epoch} batch {i} loss: {loss.item()}')
                sy = ''
                sy_pred = ''

                for k in list(map(self.dataset.get_itoc, [int(j) for j in y[0]])):
                    sy += k
                for k in list(map(self.dataset.get_itoc, [int(j) for j in y_pred_argmax[0]])):
                    sy_pred += k
                print('---------------------------------------------------') 
                print(sy)
                print(sy_pred)
                print('---------------------------------------------------')
                # validation step - TODO make better
                #if i % 500 == 499:
                #    str = self.complete(model, "ROMEO:", 128)
                #    running_loss /= 100
                #print(f'epoch {self.epoch} batch {i} loss: {running_loss}')
                #print(self.tokenizer.batch_decode(x)[0])
                #print(self.tokenizer.batch_decode(y_test)[0])
                #print(self.tokenizer.batch_decode(y_pred_argmax)[0])
                if i % 1000 == 0:
                    torch.save(model.state_dict(), f'chargpt_iter{i}.pt')
                
                #self.complete(model, 'ROMEO:', 128

    def complete(self, model, _start, length):
        _in = torch.Tensor(list(map(self.dataset.get_ctoi, _start.lower()))) \
        .type(torch.LongTensor) \
        .to(self.config.device) \
        .unsqueeze(0)   

        for i in range(length):
            _out = torch.argmax(model(_in[:, :self.config.max_seq_length]), -1)[-1:]
            print(_in.shape)
            print(_out.shape)
            _in = torch.cat([_in, _out])
        _in = list(_in[0])
        a = ''
        for i in _in:
            a += self.dataset.get_itoc(int(i))
        print(a)

class ViTTrainer:
    def __init__(self, dataset, config):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True) # map-style dataset
        self.config = config
        self.dataset = dataset

    def train(self, model, **kwargs):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, betas=(0.9, 0.98))
        loss_fn = torch.nn.CrossEntropyLoss()
        running_loss = 0
        model.train()
        for epoch in range(self.config.epochs):
            for i, (x, y) in enumerate(self.dataloader):
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                y_pred = model(x)
                model.zero_grad(set_to_none=True)
                loss = loss_fn(y_pred.view(-1, 10), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                running_loss += loss.item()
                print(f'epoch {epoch} batch {i} loss: {loss.item()}')