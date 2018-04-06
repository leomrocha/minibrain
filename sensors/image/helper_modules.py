

import numpy as np
import torch
from torch import nn
# import torchvision
# from torch.nn import functional as F
from torch.autograd import Variable
# from torchvision import transforms, utils


# If need Bilinear interpolation, take it from from here:
# https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

class SimpleMergingLayer(nn.Module):
    """
    This NN layer solves the issue of learning possible factors needed for the merging of multiple images
    The idea is that several images that span the SAME region and size might be needed to be merged,
    the issue is if those images correspond to different resolutions, the importance of each image
    is not necessary the same, this means that only doing a simple process can give suboptimal results
    This merging layers creates one image from the given number of layers N and learns a SINGLE
    weight value for each layer
    the input to this layer is a list of (same shaped) images to be merged
    the output is a single merged image
    """

    def __init__(self, n_layers):
        """
        @param in_imag_shape : [width, height]  # the input image shape, to be able to pre-compute the transform matrices
        """
        super(SimpleMergingLayer, self).__init__()
        self.n_layers = n_layers  # number of layers to merge
        # start the weight of the tensors as all the same
        self.n_weights = [Variable(torch.FloatTensor([1.0/n_layers]).cuda(), requires_grad=True) for _ in range(n_layers)]
        # self.n_weights = [Variable(torch.FloatTensor([1.0/n_layers]).cuda(), requires_grad=False) for _ in range(n_layers)]

    def forward(self, images):
        """
        :param images: list of images to merge
        :return: a single merged image
        """
        assert(len(images) > 0)
        img = images[0]
        img = img * (self.n_weights[0])
        if len(images) > 1:
            for i in range(len(images[1:])):
                nimg = images[i+1]
                img = img + (nimg * (self.n_weights[i]))

        return img


# TODO make more complex and interesting processing for the merging layers
