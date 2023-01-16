#!/usr/bin/env python
# coding: utf-8

# In[ ]:

####Implement your model in a script called model.py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
from torch.utils.data import Dataset
import wandb


class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        
        self.criterion = nn.CrossEntropyLoss() #CRITERIUM


    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
    def training_step(self, batch, batch_idx): #NEW (for LightningModule )
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        # self.logger.experiment is the same as wandb.log
        self.logger.experiment.log({'logits': wandb.Histrogram(preds)})
        return loss

    def test_step(self, batch, batch_idx):   #NEW (for LightningModule )
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=1)).float().mean()
        self.log("val_acc", acc, logger=True)
        self.log("val_loss", loss, logger=True)
        # self.logger.experiment.log({'val_logits': wandb.plot.Histrogram(preds)})
        return loss

    def configure_optimizers(self):  #NEW (for LightningModule )
        return optim.Adam(self.parameters(), lr=1e-2)



class mnist(Dataset):
    def __init__(self, train):
        test_input_filepath = 'C:/Users/usuario/MLOPS/nuevo_repo/data/processed/test.npz'
        train_input_filepath = 'C:/Users/usuario/MLOPS/nuevo_repo/data/processed/train_data.npz'
        if train:
            content = [ ]
            content.append(np.load(train_input_filepath, allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
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


def main():
    train_set = mnist(train=True)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=128)

    test_set = mnist(train=False)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    model = MyAwesomeModel()

    """
    ###Callbacks to add to the model:
    #ModelCheckpoint callback-->saving checkpoints only when some metric improves
    checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
    )

    #EarlyStopping callback-->automatically stopping the training if a certain value is not improving anymore
    early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    """


                    #callbacks=[checkpoint_callback, early_stopping_callback]
    trainer = Trainer(accelerator="gpu", 
                    max_epochs=10, 
                    max_steps=4, 
                    logger=pl.loggers.WandbLogger(project="dtu_mlops_light"),
                    limit_train_batches=0.2) 
    
    trainer.fit(model, train_dataloaders=train_data_loader) #Fit the model
    trainer.test(model, dataloaders=test_data_loader) #Test the model

if __name__ == "__main__":
    main()