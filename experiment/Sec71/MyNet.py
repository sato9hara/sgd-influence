import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class NetList(torch.nn.Module):
    def __init__(self, list_of_models):
        super(NetList, self).__init__()
        self.models = torch.nn.ModuleList(list_of_models)
    
    def forward(self, x, idx=0):
        return self.models[idx](x)

class LogReg(nn.Module):
    def __init__(self, input_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        x = self.fc(x)
        return x
        #x = torch.sigmoid(x)
        #x = torch.cat((x, 1 - x), dim=1)
        #return torch.log(x)

class DNN(nn.Module):
    def __init__(self, input_dim, m=[8, 8]):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, m[0]),
            nn.ReLU(),
            nn.Linear(m[0], m[1]),
            nn.ReLU(),
            nn.Linear(m[1], 1),
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
        #x = torch.sigmoid(x)
        #x = torch.cat((x, 1 - x), dim=1)
        #return torch.log(x)
