import sys
sys.path.append("./src/models/")

import os

import pytest
import pytorch_lightning as pl
import torch
from lightning_model import MyAwesomeModel, mnist
from pytorch_lightning import LightningModule, Trainer

def test_training(): #Training created
    model = MyAwesomeModel()
    train_set = mnist(train=True)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=128)
    
    trainer = Trainer(
        max_epochs=2,
        logger=pl.loggers.WandbLogger(project="dtu_mlops_light"),
        precision=16,
    )
    trainer.fit(model, train_dataloaders=train_data_loader) #fit the model
    logged_metrics = trainer.logged_metrics #metrics from the fitted model
    assert logged_metrics["train_acc"] >= 0, "Train accuracy less than 0"
    assert logged_metrics["train_acc"] <= 1, "Train accuracy smaller than 1"


test_training()