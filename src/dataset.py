import argparse

import numpy as np
import os
import os.path
import torch
import warnings

import torchvision.transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
from typing import Any, Callable, Dict, Optional, Tuple
from torch.utils import data
from torchvision import datasets, transforms

from src.utils.util import chunkify
from torch.utils.data import DataLoader
#from src.FedCBO import get_dataloaders


class rotatedMNIST:

    def __init__(self, args, train, download, transform, cluster_idx, is_subset=False):
        self.args = args
        self.train = train
        self.transform = transform
        self.cluster_idx = cluster_idx

        self.dataset = datasets.MNIST(root=self.args.data_path, train=self.train,
                                      transform=self.transform, download=download)
        self.targets = self.dataset.targets
        self.rotate()
        if is_subset:
            self.get_subset()

    def rotate(self):
        if self.args.p == 4:
            k = self.cluster_idx
        elif self.args.p == 2:
            k = (self.cluster_idx % 2) * 2
        elif self.args.p == 1:
            k = 0
        else:
            raise NotImplementedError('only p=1,2,4 supported.')
        
        # normalize to have 0 ~ 1 range in each pixel
        self.dataset.data = self.dataset.data / 255.0

        self.data = torch.rot90(self.dataset.data, k=int(k), dims=(1,2))
        self.data = self.data.reshape(-1, 28*28)

    def get_subset(self):
        # Create a subset of original dataset
        random_seed = 543 # Same seed for same subset picking every time
        np.random.seed(random_seed)
        data_idx = np.arange(len(self.targets))
        subset_idx = np.random.choice(data_idx, int(0.1 * len(self.targets)))
        self.data = self.data[subset_idx]
        self.targets = self.targets[subset_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)








if __name__ == '__main__':
    #test_hetero_mnist()
    # test_rotated_mnist()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument("--p", type=int, default=4, help='Number of clusters.')
    parser.add_argument('--N', type=int, default=4, help='Number of agents.')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    transform = transforms.ToTensor()

    dataset = rotatedMNIST(args, train=True, download=True, transform=transform, cluster_idx=1, is_subset=False)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    data = list(dataset)[:100]

    for k, (image, label) in enumerate(data[:9]):
        plt.subplot(3,3,k+1)
        plt.imshow(image.numpy().reshape(28,28), cmap='gray')
        plt.show()










