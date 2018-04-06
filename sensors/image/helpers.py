
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision import datasets
from torchvision.utils import save_image

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as st


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return sorted(factors)


def tensor_to_img(x, width, height, channels):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), channels, width, height)
    return x


# From https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

# TODO improve this to make it from Standard Deviation and kernel size
def get_gaussian_kernel(kernlen=5, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


# definitions of the operations for the full image autoencoder
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],  # from example here https://github.com/pytorch/examples/blob/409a7262dcfa7906a92aeac25ee7d413baa88b67/imagenet/main.py#L94-L95
   std=[0.229, 0.224, 0.225]
   # mean=[0.5, 0.5, 0.5], # from example here http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
   # std=[0.5, 0.5, 0.5]
)


# the whole image gets resized to a small image that can be quickly analyzed to get important points
def monochrome_preprocess(w=48, h=48):
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((w, h)),  # this should be used ONLY if the image is bigger than this size
        transforms.ToTensor(),
        normalize
    ])


def fullimage_preprocess(w=48, h=48):
    return transforms.Compose([
        transforms.Resize((w, h)),  # this should be used ONLY if the image is bigger than this size
        transforms.ToTensor(),
        normalize
    ])


# the full resolution fovea just is a small 12x12 patch
def crop_fovea(size=12):
    sample = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize
    ])
    return sample


def downsample_tensor(crop_size, final_size=16):
    sample = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(final_size),
        transforms.ToTensor(),
        normalize
    ])
    return sample


def get_loaders(batch_size, transformation, dataset=datasets.CIFAR100, cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset('../data', train=True, download=True,
                       transform=transformation),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset('../data', train=False, transform=transformation),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader
