import torch
from tqdm import tqdm

# TODO makek the two trainers subclass from a shared Trainer class for OOP points

class Trainer:
    def __init__(self, dataset, config):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True) # map-style dataset
        self.config = config
        self.dataset = dataset

    def train(self, model, optimizer, **kwargs):

        loss_fn = torch.nn.CrossEntropyLoss()
        model.train()
        running_loss = 0
 
        # this works for chargpt, but needs to be generalized for models that need both x, y as inputs, like translators
        for epoch in range(self.config.epochs):
            for i, (x, y) in enumerate(self.dataloader):

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


                if i % 500 == 0:
                    # evaluate both the train and test score
                    model.eval()
                    with torch.no_grad():
                        # sample from the model...
                        context = "O God, O God!"
                        x = torch.tensor([self.dataset.ctoi[s] for s in context], dtype=torch.long)[None,...].to(self.config.device)
                        y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                        completion = ''.join([self.dataset.itoc[int(i)] for i in y])
                        print(completion)
                    
                    model.train()
                
                if i % 1000 == 0:
                    torch.save(model.state_dict(), f'chargpt_iter{i}.pt')

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
                print(f'epoch {epoch} batch {i} loss: {loss.item()}')

