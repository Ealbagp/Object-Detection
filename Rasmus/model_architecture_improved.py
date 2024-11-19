import torch
import torch.nn as nn
import torch.nn.functional as F

init_type = init=nn.init.xavier_uniform_


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomCNN(nn.Module):
    def __init__(self, proposal_size=(64, 64), num_classes=2, base_channels=64):
        super(CustomCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = self._make_layer(base_channels, base_channels, stride=1)
        self.layer3 = self._make_layer(base_channels, base_channels * 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 2, base_channels * 4, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class NetworkImproved(nn.Module):
    def __init__(self, proposal_size=(64, 64), base_channels=64):
        super(NetworkImproved, self).__init__()
        self.convolutional1 = nn.Sequential(
                nn.Conv2d(3, base_channels, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),  # Extra layer
                nn.BatchNorm2d(base_channels, eps=1e-04),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
        )

        self.convolutional2 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),  # Extra layer
                nn.BatchNorm2d(base_channels * 2, eps=1e-04),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
        )
        
        self.convolutional3 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),  # Extra layer
                nn.BatchNorm2d(base_channels * 4, eps=1e-04),
                nn.ReLU(inplace=True),
        )

        self.fully_connected = nn.Sequential(
                nn.Linear((proposal_size[0] // 4) * (proposal_size[1] // 4) * base_channels * 4, 512),
                nn.BatchNorm1d(512, eps=1e-04),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, eps=1e-04),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
                )
        
    def forward(self, x):
        x = self.convolutional1(x)
        x = self.convolutional2(x)
        x = self.convolutional3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fully_connected(x)
        
        return x
