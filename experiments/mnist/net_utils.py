
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torch.optim as optim
from torchvision import datasets, transforms


def get_loaders(batch_size, transformation, dataset=datasets.MNIST, cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(dataset('../data', train=True, download=True, transform=transformation),
                              batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset('../data', train=False, transform=transformation),
                             batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader
