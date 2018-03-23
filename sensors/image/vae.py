import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision import datasets
from torchvision.utils import save_image
import skimage 

# import io
# import requests
# from PIL import Image

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import os

# These classes are mostly taken and modified from the official examples here:
# https://github.com/pytorch/examples/blob/master/vae/main.py 
# and another here: https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py

class VAEEncoder(nn.Module):
    """
    The Encoder = Q(z|X) for the Network
    As a Variational AutoEncoder with internal linear units
    """
    def __init__(self, in_dim, hid_dim, out_dim):
        super(VAEEncoder, self).__init__()
        #hid_dim = out_dim*20
        #in_dim = w * h * channels
        self.fc1 = nn.Linear(in_dim, hid_dim)
        #self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc21 = nn.Linear(hid_dim, out_dim)
        self.fc22 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()
        #self.relu2 = nn.ReLU()
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu    
    
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        #h2 = self.relu2(self.fc2(h1))
        mu, logvar = self.fc21(h1), self.fc22(h1)
        #mu, logvar = self.fc21(h2), self.fc22(h2)
        return self.reparameterize(mu, logvar), mu, logvar


class VAEDecoder(torch.nn.Module):
    """
    The Decoder = P(X|z) for the Network
    As a Variational AutoEncoder with internal linear units
    """
    def __init__(self, in_dim, hid_dim, out_dim):
        super(VAEDecoder, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, hid_dim)
        self.linear2 = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.sigmoid(self.linear2(x))

    
class VAE(nn.Module):
    def __init__(self, width, height, channels, h_dim=400, code_dim=100):
        super(VAE, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.img_dim = width*height*channels
        self.encoder = VAEEncoder(self.img_dim, h_dim, code_dim)
        self.decoder = VAEDecoder(code_dim, h_dim, self.img_dim)
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x.view(-1, self.img_dim))
        return self.decoder(z), mu, logvar
        
    def save_model(self, name, path):
        torch.save(self.encoder, os.path.join(path, "vae_encoder_"+name+".pth"))
        torch.save(self.decoder, os.path.join(path, "vae_decoder_"+name+".pth"))
        

#definitions of the operations for the full image autoencoder
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406], # from example here https://github.com/pytorch/examples/blob/409a7262dcfa7906a92aeac25ee7d413baa88b67/imagenet/main.py#L94-L95
   std=[0.229, 0.224, 0.225]
#   mean=[0.5, 0.5, 0.5], # from example here http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#    std=[0.5, 0.5, 0.5]
)
#test one:
preprocess = transforms.Compose([
    transforms.Resize(50),
    transforms.ToTensor()
])

#the whole image gets resized to a small image that can be quickly analyzed to get important points
def fullimage_preprocess(w,h):
    transf = transforms.Compose([
        transforms.Resize((w,h)), #this should be used ONLY if the image is bigger than this size
        transforms.ToTensor(),
        normalize
    ])
    return transf

#the full resolution fovea just is a small 10x10 patch 
full_resolution_crop = transforms.Compose([
    transforms.RandomCrop(10),
    transforms.ToTensor(),
    normalize
])

def downsampleTensor(crop_size, final_size=15):
    sample = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.Resize(final_size), 
        transforms.ToTensor(),
        normalize
    ])
    return sample
    
# #the first downsampled filter will have 3 times the spanned range of the fovea, 
# # but will be downsampled to half the definition
# downsample_1_crop = downsampleTensor(30)
# #the second downsampled filter will have 6 times the spanned range of the fovea,
# # but will be downsampled to half of the previous downsampled the definition (or 1/4 resolution of the fovea)
# downsample_2_crop = downsampleTensor(60)
# #the first downsampled filter will have 12 times the spanned range of the fovea,
# # but will be downsampled to half of the previous downsampled the definition (or 1/8 resolution of the fovea)
# downsample_3_crop = downsampleTensor(120)


def get_loaders(batch_size, transformation, dataset = datasets.CIFAR100, cuda=True):

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset('../data', train=True, download=True,
                       transform=transformation),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset('../data', train=False, transform=transformation),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def train(model, optimizer, loss_function, train_loader, epoch, batch_size, width, height, channels, log_interval=100, cuda=True):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, width, height, channels)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    

def test(name, model, test_loader, epoch, batch_size, width, height, channels, cuda=True):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        if cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, width, height, channels).data[0]
        if i == 0:
            n = min(data.size(0), 100)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(batch_size, channels, width, height)[:n]])
            save_image(comparison.data.cpu(),'vae_results/' + name + '_reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    
def train_save_model(name, model, optimizer, loss_function, transformation, width, height, channels, epochs=10, batch_size=128, dataset = datasets.CIFAR100, cuda=True):
    

    train_loader, test_loader = get_loaders(batch_size, transformation)

    for epoch in range(1, epochs + 1):
        train(model, optimizer, loss_function, train_loader,epoch, batch_size, width, height, channels)
        test(name, model, test_loader,epoch, batch_size, width, height, channels)

    #torch.save(model, name+".pth")
    model.save_model(name, "VAE")
    
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, width, height, channels):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, width*height*channels), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train_all(models, epochs=10):
    for n,m,t in models:
        print("--------------------------------------------------")
        print("training model "+ n)
        m.cuda()
        optimizer = optim.Adam(m.parameters(), lr=1e-3)
        train_save_model(n,m, optimizer, loss_function, t, m.width, m.height, m.channels, dataset = datasets.CocoDetection, epochs=epochs)

        
if __name__ == "__main__":
    #numbers are chosen so the image size allows for at least 2 convolution layers in a CNN (they'll be used for other parts)
    models = [
          ("fovea_12x12", VAE(12,12,3,500,100),full_resolution_crop),
          ("downsample1_16x16-30x30", VAE(16,16,3,500,100), downsampleTensor(30,16)),
          # commented due to image size, it seems that some images fail the training, I'll need to find a good dataset
          # ("downsample2_16x16-60x60", VAE(16,16,3,500,100), downsampleTensor(60,16)),
          # ("downsample3_16x16-120x120", VAE(16,16,3,500,100), downsampleTensor(120,16)),
          # ("downsample4_20x20-240x240", VAE(20,20,3,500,100), downsampleTensor(240,20)),
          # ("downsample5_20x20-480x480", VAE(20,20,3,500,100), downsampleTensor(480,20)),
          ("fullimage_20", VAE(20,20,3,500,100), fullimage_preprocess(20,20)),
          ("fullimage_48", VAE(48,48,3,500,100), fullimage_preprocess(48,48)),
          ("fullimage_72", VAE(72,72,3,500,100), fullimage_preprocess(72,72)),
          ("fullimage_100", VAE(96,96,3,500,100), fullimage_preprocess(96,96)),
         ]
    train_all(models)