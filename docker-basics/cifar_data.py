import pickle
import torch
from torch.utils.data import Dataset


class CIFAR10(Dataset):
    def __init__(self, root, test=False):
        self.root = root
        self.paths = [
            "cifar-10/data_batch_1",
            "cifar-10/data_batch_2",
            "cifar-10/data_batch_3",
            "cifar-10/data_batch_4",
            "cifar-10/data_batch_5"
        ]
        self.test_path = ["cifar-10/test_batch"]
        self.test = test
    
    def __len__(self):
        if self.test is False:
            return 50000
        else:
            return 10000

    def unpickle(self, file):
        with open(file, "rb") as f:
            dict = pickle.load(f, encoding="bytes")
            labels = dict[b"labels"]
            data = dict[b"data"]
        return data, labels

    def load_data(self):
        self.dataset_all = []
        self.labels_all = []
        if self.test is False:
            for i in self.paths:
                data, labels = self.unpickle(self.root + i)
                self.dataset_all.extend(data)
                self.labels_all.extend(labels)
        else:
            self.dataset_all, self.labels_all = self.unpickle(self.root + self.test_path[0])


    def __getitem__(self, index):
        img = self.dataset_all[index]
        label = self.labels_all[index]
        img = img.reshape(3, 32, 32).astype('float32')
        return img, label
