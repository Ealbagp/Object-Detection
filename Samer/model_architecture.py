import torch
import torch.nn as nn

init_type = init=nn.init.xavier_uniform_

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional1 = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
        )

        self.convolutional2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256,eps=1e-04),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
        )
        
        self.convolutional3 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128,eps=1e-04),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
        )
        self.convolutional4 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128,eps=1e-04),
                nn.ReLU(inplace=True),
        )
        

        self.fully_connected = nn.Sequential(
                nn.Linear(8 * 8 * 128, 1024), # 64 by 3 convolutions = (((64 / 2) / 2) / 2) = 8
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Sigmoid()
                # nn.Softmax(dim=1)
                )
        
    def forward(self, x):
        x = self.convolutional1(x)
        x = self.convolutional2(x)
        x = self.convolutional3(x)
        x = self.convolutional4(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.fully_connected(x)
        
        return x
