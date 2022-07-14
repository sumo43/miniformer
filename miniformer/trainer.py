import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, dataset, config):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) # map-style dataset
        self.config = config

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
                if i % 50 == 99:
                    running_loss /= 100
                    print(f'epoch {self.epoch} batch {i} loss: {running_loss}')
                    running_loss = 0
                """ 
                # validation loop/step
                if i % 500 == 99:
                    running_loss /= 100
                    print(f'epoch {self.epoch} batch {i} loss: {running_loss}')
                    #print(self.tokenizer.batch_decode(x)[0])
                    #print(self.tokenizer.batch_decode(y_test)[0])
                    #print(self.tokenizer.batch_decode(y_pred_argmax)[0])
                """

            model.save(os.path.join(WEIGHTS_DICT, f'chargpt_epoch{epoch}.pt'))
        
        def validate(self, model, **kwargs):
            pass