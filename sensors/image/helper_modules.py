
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms


class DownsamplerLayer(nn.Module):
    """
    """
    def __init__(self, width, height):
        super().__init__()
        self.resizer = transforms.Compose([
                        transforms.ToPILImage(),  #is this correct? will this be slow??
                        transforms.Resize(width, height),
                        transforms.ToTensor()
                        ])
    def forward(self, x):
        return self.resizer(x)
        #return self.conv(x)
