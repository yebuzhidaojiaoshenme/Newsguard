import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(FusionModel, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        fused = torch.tanh(text_proj + image_proj)
        return self.fc(fused)
