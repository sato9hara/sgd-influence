import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MnistNet(torch.nn.Module):
    def __init__(self, m=[20, 20]):
        super(MnistNet, self).__init__()
        self.m = m
        self.conv1 = torch.nn.Conv2d(1, self.m[0], 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(self.m[0], self.m[1], 5, stride=1, padding=0)
        self.fc = torch.nn.Linear(4*4*self.m[1], 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.m[1])
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=1)
    
    def flatten(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.m[1])
        return x

class CifarNet(nn.Module):
    def __init__(self, m=[32, 32, 64, 64, 128, 128]):
        super(CifarNet, self).__init__()
        self.conv_layer = nn.Sequential(
            
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=m[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[0], out_channels=m[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer block 2
            nn.Conv2d(in_channels=m[1], out_channels=m[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[2], out_channels=m[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer block 3
            nn.Conv2d(in_channels=m[3], out_channels=m[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[4], out_channels=m[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(16*m[5], 10),
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return torch.nn.functional.log_softmax(x, dim=1)
    
    def flatten(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x
    
class NetList(torch.nn.Module):
    def __init__(self, list_of_models):
        super(NetList, self).__init__()
        self.models = torch.nn.ModuleList(list_of_models)
    
    def forward(self, x, idx=0):
        return self.models[idx](x)

class MnistAE(nn.Module):
    def __init__(self, device, m=[24, 12]):
        super(MnistAE, self).__init__()
        self.m = m
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.m[0], 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(self.m[0], self.m[1], 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1, padding=0)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.m[1], self.m[1], 5, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.m[1], self.m[0], 4, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.m[0], 1, 3, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class CifarAE(nn.Module):
    def __init__(self, device, m=[64, 32, 16]):
        super(CifarAE, self).__init__()
        self.m = m
        self.mm = np.array((0.4914, 0.4822, 0.4465))[np.newaxis, :, np.newaxis, np.newaxis]
        self.ss = np.array((0.2023, 0.1994, 0.2010))[np.newaxis, :, np.newaxis, np.newaxis]
        self.mm = torch.from_numpy(self.mm).float().to(device)
        self.ss = torch.from_numpy(self.ss).float().to(device)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.m[0], 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(self.m[0], self.m[1], 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1, padding=0)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.m[1], self.m[1], 5, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.m[1], self.m[0], 4, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.m[0], 3, 3, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = 0.5 * (x + 1)
        x = (x - self.mm) / self.ss
        return x