import torch
import click
# import matplotlib.pyplot as plt
import numpy as np

from model import MyAwesomeModel #from the file 'model.py' we have in the same folder
from torch import nn, optim
from torch.utils.data import Dataset
import wandb


@click.group()
def cli():
    pass

# Added commentary line to try new Branch
@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    wandb.init(project="mi_proyecto")  #M13 Experiment logging
    print("Training day and night")
    print(lr) #learning rate

    model = MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    train_set = mnist(train=True)
    print('antes')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    
    n_epoch = 2
    for epoch in range (n_epoch):
        loss_tracker = []
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            print(log_ps)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())


        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
        wandb.log({"train_loss": loss_tracker}) #M13 Experiment logging

    torch.save(model.state_dict(), 'C:/Users/usuario/MLOps/nuevo_repo/src/models/trained_model.pt')
    
    image = wandb.Image(loss_tracker, caption=f"random field") #M13 Experiment logging
    # plt.plot(loss_tracker, '-')
    # plt.xlabel('Training step')
    # plt.ylabel('Training loss')
    # plt.savefig("reports/figures/training_curve.png")


def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    test_set = mnist(train=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    with torch.no_grad():
        for images, labels in testloader:
            ps = torch.exp(model(images)) #probabilities of each class
            top_p, top_class = ps.topk(1, dim=1) #class with the hihger probabilty
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Accuracy: {accuracy.item()*100}%')
            wandb.log({"va_accuracy": accuracy.item()})  #M13

class mnist(Dataset):
    def __init__(self, train):
        test_input_filepath = 'C:/Users/usuario/MLOps/nuevo_repo/data/processed/test.npz' #where TEST data processed is
        train_input_filepath = 'C:/Users/usuario/MLOps/nuevo_repo/data/processed/train_data.npz' #where TRAIN data processed is
        if train: #TRAIN data
            content = [ ]
            content.append(np.load(train_input_filepath, allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))

            print(f'data shape: {data.shape}')
            print(f'data shape: {data.shape}')


        else: #TEST data
            content = [ ]
            content = np.load(test_input_filepath, allow_pickle=True)
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
            
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]

cli.add_command(train)
if __name__ == "__main__":
    cli()
