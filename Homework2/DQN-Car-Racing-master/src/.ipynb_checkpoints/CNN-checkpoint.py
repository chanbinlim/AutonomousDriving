import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.LayerNorm2d1 = nn.LayerNorm([16, 20, 20])
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.LayerNorm2d2 = nn.LayerNorm([32, 9, 9])
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # [N, 32, 9, 9] -> [N, 64, 7, 7]
        self.LayerNorm2d3 = nn.LayerNorm([64, 7, 7])
        
        self.in_features = 64 * 7 * 7
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        # Conv layer 1
        x = self.conv1(x)
        x = self.LayerNorm2d1(x)
        x = self.activation(x)
        
        # Conv layer 2
        x = self.conv2(x)
        x = self.LayerNorm2d2(x)
        x = self.activation(x)

        # conv layer 3
        x = self.conv3(x)
        x = self.LayerNorm2d3(x)
        x = self.activation(x)
        
        x = x.view((-1, self.in_features))
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x