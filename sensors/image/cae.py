import torch
from torch import nn, optim
from torch.nn import functional as F

import os

from helpers import *

###
# Mostly Taken from examples here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://github.com/csgwon/pytorch-deconvnet/blob/master/models/vgg16_deconv.py
# Other resources
# https://github.com/pgtgrly/Convolution-Deconvolution-Network-Pytorch/blob/master/conv_deconv.py
# https://github.com/kvfrans/variational-autoencoder
# https://github.com/SherlockLiao/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main-gpu.py
# https://pgaleone.eu/neural-networks/2016/11/24/convolutional-autoencoders/
###

class CAEEncoder(nn.Module):
    """
    The Encoder = Q(z|X) for the Network
    """

    def __init__(self, width, height, channels=3, levels=2, kernel_size=3, first_feature_count=16):
        super(CAEEncoder, self).__init__()
        # print("CAE init ", width, height, channels, levels, kernel_size, first_feature_count)
        self.width = width
        self.height = height
        self.channels = channels
        # compute the maximum number of levels that this resolution can handle,
        # this will be the parameter given to create the resolution encoder
        max_levels = prime_factors(min(width, height)).count(2)
        self.levels = min(levels, max_levels)
        self.kernel_size = kernel_size
        self.first_feature_count = first_feature_count

        self.indices = []

        padding = kernel_size // 2

        self.l_features = [channels]
        self.layers = nn.ModuleList()

        for i in range(self.levels + 1):
            self.l_features.append(first_feature_count * (2 ** (i)))

        for i in range(self.levels):
            nfeat = self.l_features[i + 1]
            layer = nn.Sequential(
                nn.Conv2d(self.l_features[i], nfeat, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.Conv2d(nfeat, nfeat, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2, return_indices=True)
            )
            self.layers.append(layer)

        self.conv_dim = ((width * height) // ((2 ** levels) ** 2)) * self.l_features[-1]

    def forward(self, x):
        self.indices = []
        out = x
        for i in range(self.levels):
            layer = self.layers[i]
            out, idx = layer(out)
            self.indices.append(idx)
        return out


class CAEDecoder(torch.nn.Module):
    """
    The Decoder = P(X|z) for the Network
    """

    def __init__(self, encoder, width, height, channels=3, levels=2, kernel_size=3, first_feature_count=16):
        super(CAEDecoder, self).__init__()
        padding = kernel_size // 2
        self.width = width
        self.height = height
        self.channels = channels
        max_levels = prime_factors(min(width, height)).count(2)
        self.levels = min(levels, max_levels)
        self.encoder = encoder

        self.l_features = [channels]
        self.layers = nn.ModuleList()

        for i in range(self.levels + 1):
            self.l_features.append(first_feature_count * (2 ** i))

        self.encoder = encoder
        self.conv_dim = ((width * height) // ((2 ** levels) ** 2)) * self.l_features[-1]

        for i in range(self.levels):
            nfeat = self.l_features[i + 1]

            last_op = nn.ReLU() if i + 1 >= self.levels else nn.Tanh()  # the last operation is the one in the first layer -> later is reversed
            layer = nn.Sequential(
                nn.ConvTranspose2d(nfeat, nfeat, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.ConvTranspose2d(nfeat, self.l_features[i], kernel_size=kernel_size, padding=padding),
                last_op

            )
            self.layers.append(layer)
        self.layers = self.layers[::-1]  # reverse the layers, all is processed at the reverse of the encoders

    def forward(self, x):
        out = x
        for i in range(self.levels):
            rev_i = -(i + 1)
            out = F.max_unpool2d(out, self.encoder.indices[rev_i], 2, stride=2)
            out = self.layers[i](out)
        return out


class CAE(nn.Module):
    def __init__(self, width, height, channels, levels=2, conv_layer_feat=16):
        super(CAE, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.encoder = CAEEncoder(width, height, channels, levels, 3, conv_layer_feat)
        self.decoder = CAEDecoder(self.encoder, width, height, channels, levels, 3, conv_layer_feat)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def save_model(self, name, path):
        torch.save(self.encoder, os.path.join(path, "cae_encoder_" + name + ".pth"))
        torch.save(self.decoder, os.path.join(path, "cae_decoder_" + name + ".pth"))

