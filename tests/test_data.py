import sys
sys.path.append("./src/models/") #to read lightning_model module

import os
import pytest
import torch
from lightning_model import mnist #src/models/lighting_model.py ###from the file lighting

#Check if test data exists
import os.path
@pytest.mark.skipif(not os.path.exists("data/processed/test.npz"), reason="Test data files not found")

#Check if train data exists
@pytest.mark.skipif(not os.path.exists("data/data/processed/train_data.npz"), reason="Train data files not found")


#@pytest.mark.parametrize("train", [True, False])  ####To check a function/module for various input arguments
def test_data():
    dataset = mnist(train=True)

    ####assert that len(dataset) == N_train for training and N_test for test
    if True ==True: #if we are checking training data
        N_data = 25000 
    else:            #if we are checking testing data
        N_data = 5000
    assert len(dataset) == N_data, "Dataset do not have the correct number of samples"

    ####assert that each datapoint has shape [1,28,28]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for image, label in data_loader:
        assert image.shape == torch.Size([1,1,28,28]), "Datapoints do not have the correct shape"
    


test_data()