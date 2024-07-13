import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(model, dataloader, num_epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data in dataloader:
            text_features = data['text']
            image_features = data['image']
            labels = data['label']

            optimizer.zero_grad()
            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * text_features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Training complete')
    return model
