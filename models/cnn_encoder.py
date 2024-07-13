import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNEncoder, self).__init__()
        self.cnn = models.resnet50(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

    def forward(self, images):
        features = self.cnn(images)
        features = features.view(features.size(0), -1)
        return features
