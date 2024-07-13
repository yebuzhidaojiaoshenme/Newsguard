import torch
import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, input_dim):
        super(Detector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
