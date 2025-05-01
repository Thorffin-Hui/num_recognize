import torch
import torch.nn as nn

params = torch.load('./mymodel.py')

model = nn.Sequential(
    nn.Linear(784,444),
    nn.ReLU(),
    nn.Linear(444,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,10),
    nn.Softmax(),
).cuda()

model.load_state_dict(params)

