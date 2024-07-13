import torch
from torch.utils.data import Dataset

class NeuralNewsDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'text': sample['text'],
            'image': sample['image'],
            'caption': sample['caption'],
            'label': sample['label']
        }


def get_dataloader(data_file, batch_size=32, shuffle=True):
    dataset = NeuralNewsDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
