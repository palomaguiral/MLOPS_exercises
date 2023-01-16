import sys
sys.path.append("./src/models/")

import torch
from lightning_model import MyAwesomeModel


# a
def test_model():
    model = MyAwesomeModel()
    x = torch.zeros(1, 1, 28, 28)  #data (example)
    y = model(x) #apply the model

    #Check the size of the INPUT data:
    assert x.shape == torch.Size(  
        [1, 1, 28, 28]
    ), "Input size is not torch.Size([1, 1, 28, 28])"

    #Check the size of the OUTPUT data
    assert y.shape == torch.Size([1, 10]), "Output size is not torch.Size([1,10])"


test_model()