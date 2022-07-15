import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, dataset, config):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True) # map-style dataset
        self.config = config
        self.dataset = dataset

    def train(self, model, **kwargs):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        running_loss = 0
        # this works for chargpt, but needs to be generalized for models that need both x, y as inputs, like translators
        for epoch in range(self.config.epochs):
            for i, (x, y) in enumerate(self.dataloader):
                y_pred = model(x)
                #y_pred_argmax = torch.argmax(y_pred, -1)
                optimizer.zero_grad()
                loss = loss_fn(y_pred.view(-1, self.config.vocab_size), y.view(-1,))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # running loss
                if i % 50 == 49:
                    running_loss /= 50
                    print(f'epoch {self.config.epochs} batch {i} loss: {running_loss}')
                    running_loss = 0
                
                # validation step - TODO make better
                if i % 500 == 499:
                    str = self.complete(model, "ROMEO:", 128)
                    running_loss /= 100
                    #print(f'epoch {self.epoch} batch {i} loss: {running_loss}')
                    #print(self.tokenizer.batch_decode(x)[0])
                    #print(self.tokenizer.batch_decode(y_test)[0])
                    #print(self.tokenizer.batch_decode(y_pred_argmax)[0])
                
            model.save(os.path.join(WEIGHTS_DICT, f'chargpt_epoch{epoch}.pt'))
        
    def complete(self, model, _start, length):
        _in = torch.Tensor(list(map(self.dataset.get_ctoi, _start.lower()))).type(torch.LongTensor)
        for i in range(length):
            _out = torch.argmax(model(_in), -1)[-1:]
            _in = torch.cat([_in, _out])
        _in = list(_in)
        a = ''
        for i in _in:
            a += self.dataset.get_itoc(int(i))
        print(a)