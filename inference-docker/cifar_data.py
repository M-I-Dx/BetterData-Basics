import pickle
import torch
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = pickle.load(open(path, 'rb'), encoding='bytes')
        self.data = self.dataset[b'data']
        self.labels = self.dataset[b'labels']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        img = img.reshape(3, 32, 32).astype('float32')
        label = self.labels[idx]
        return img, label

