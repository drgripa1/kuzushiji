import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset


class KMNISTDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        if train:
            np_images = np.load('dataset/kmnist-train-imgs.npz')['arr_0']
            np_labels = np.load('dataset/kmnist-train-labels.npz')['arr_0']
        else:
            np_images = np.load('dataset/kmnist-test-imgs.npz')['arr_0']
            np_labels = np.load('dataset/kmnist-test-labels.npz')['arr_0']

        self.length = np_labels.shape[0]
        self.images = torch.from_numpy(np_images).view(self.length, 1, 28, 28).float()
        self.images = (self.images - 128.) / 128.
        self.labels = torch.from_numpy(np_labels).long()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.length


def get_dataloader(is_train, batch_size):
    return DataLoader(
        KMNISTDataset(is_train),
        batch_size=batch_size,
        shuffle=is_train,
    )
