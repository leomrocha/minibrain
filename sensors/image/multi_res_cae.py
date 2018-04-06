

import numpy as np
import torch
from torch import nn, optim
# import torchvision
# from torch.nn import functional as F
# from torch.autograd import Variable
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms, utils
# from torchvision import datasets
# from torchvision.utils import save_image

from cae import *
from helpers import *
from helper_modules import *


class MultiResCAE(nn.Module):
    """
    Multi Resolution group of Convolutional Autoencoders
    This module intends to group several autoencoders that accompany different resolutions
    The goal of this module is be able to train and maintain all the filters in one place
    This model can be saved and loaded as a single element
    the FULL IMAGE is not considered in this, only the foveated multi resolution
    """
    # TODO This model outputs tensor of dimension 1x1xN that is the concatenation of the output of all the encoders ensemble

    def __init__(self, in_img_shape, channels=3, conv_layer_feat=[16, 16, 32],
                 res_px=[[20, 20], [16, 16], [12, 12]], crop_sizes=[[64, 64], [32, 32],  [12, 12]],
                 # conv_sizes = [(3,5,7), (3,5,7,11), (3,5,7,11)]  # this is too much I think
                 conv_sizes=[[3, 5], [3, 5], [3, 5, 7]]
                 ):
        """
        @param in_imag_shape : [width, height]  # the input image shape, to be able to pre-compute the transform matrices
        """
        super(MultiResCAE, self).__init__()
        self.channels = channels  # number of channels in the input image
        # TODO refactor, this shapes and numbers should be detected and checked here
        assert (len(res_px) == len(crop_sizes))
        self.res_levels = len(res_px)  # number of resolution levels (NOT including the full image)
        self.conv_layer_feat = conv_layer_feat  # number of convolutional filters per CAE in the first level
        self.res_px = res_px  # downsampled resolution in pixels for each resolution
        self.conv_sizes = conv_sizes  # conv filter sizes per layer, one encoder per size per layer
        ##
        # compute the maximum number of levels that this resolution can handle,
        # this will be the parameter given to create the resolution encoder
        self.max_levels = [prime_factors(min(i)).count(2) for i in res_px]
        ##
        # Pre-computing cropping matrices
        self.crop_sizes = torch.IntTensor(crop_sizes)  # Ps - Patches sizes -  size of the patch to crop
        # Ps/2 - Patches half sizes -  half size of the patch to crop, to compute positions
        self.half_crop_sizes = torch.IntTensor(np.array(crop_sizes) // 2)
        cvs = crop_sizes[:-1]  # take out the smaller size, as is the reference to the smaller patch
        cvs = [in_img_shape] + cvs
        self.ref_patch = torch.IntTensor(cvs)  # RP - Reference Patches
        # pre-compute Patch Dynamic Range (pixel wise)
        self._pdr = self.ref_patch - self.crop_sizes

        # print(self.crop_sizes, self.half_crop_sizes, self.ref_patch, self._pdr)
        # saves the last patch centers
        self._last_centers = None  # make a variable placeholder here
        # saves the last patch ranges
        self._last_px_mins = None
        self._last_px_maxs = None
        ##
        # Actual work is done in the following modules
        #
        self.downsamplers = [nn.AdaptiveAvgPool2d((w, h)) for w, h in res_px]
        self.upsamplers = [nn.AdaptiveMaxPool2d((w, h)) for w, h in crop_sizes]
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # separated as functions to be able to later LOAD the encoders instead of creating them each time
        self._create_encoders()
        self._create_decoders()

    def _create_encoders(self):
        for i in range(self.res_levels):
            res_encoders = nn.ModuleList()
            # Conv Encoder description
            conv_features = self.conv_layer_feat[i]
            l_conv_sizes = self.conv_sizes[i]
            levels = self.max_levels[i]
            # Image size
            c = self.channels
            w, h = self.res_px[i]  # resolution of the image for the encoder

            for j in range(len(l_conv_sizes)):
                ks = l_conv_sizes[j]
                enc = CAEEncoder(w, h, c, levels, ks, conv_features)
                res_encoders.append(enc)
            self.encoders.append(res_encoders)

    def _create_decoders(self):
        for i in range(self.res_levels):
            res_decoders = nn.ModuleList()
            # Conv Encoder description
            conv_features = self.conv_layer_feat[i]
            l_conv_sizes = self.conv_sizes[i]
            levels = self.max_levels[i]
            # Image size
            c = self.channels
            w, h = self.res_px[i]  # resolution of the image for the encoder

            for j in range(len(l_conv_sizes)):
                ks = l_conv_sizes[j]
                enc = CAEDecoder(self.encoders[i][j], w, h, c, levels, ks, conv_features)
                res_decoders.append(enc)
            self.decoders.append(res_decoders)

    def compute_patches(self, crop_centers):
        """
        Computes the ranges that have to be cropped from the input image.
        The computation follows the principles:
         - The inner patches positions are relative to the outer ones, in the corresponding hierarchy
         - The patches can not go off the container image

        @param crop_centers MUST be a FloatTensor or FloatTensor
        @returns : (centers, min_points, max_points) two IntTensors with the center and ranges that each patch occupies

        The returned elements contain the patches from the bigger to the smaller one
        """
        # formula follows the algorithm (but implemented in vector operations and a couple of optimizations) :
        # x_min = 0 + patch_width/2
        # x_max = full_img.width - patch_width/2
        # x_pos = x_center * (x_max - x_min) + (patch_width/2)
        # patch = img[x_pos - patch_width/2 : x_pos + patch_width/2] (warning on patch size)

        self._last_centers = crop_centers * self._pdr.float()  # + self.half_crop_sizes # element wise multiplication
        self._last_px_mins = self._last_centers.int()  # - self.half_crop_sizes
        self._last_px_maxs = self._last_centers.int() + self.crop_sizes  # + self.half_crop_sizes
        # print(self._last_centers)
        # print(self._last_px_mins)
        # print(self._last_px_maxs)

        return self._last_centers, self._last_px_mins, self._last_px_maxs

    def encode(self, x):
        full_img = x
        ########################
        # BEGIN Cropping
        ###
        # compute the patches positions
        min_px, max_px = self._last_px_mins, self._last_px_maxs
        # make only one vector out of the needed ones
        ranges = torch.cat([min_px, max_px], dim=1)
        # Get all the cropped layers
        crops = []
        # TODO find out how to do this without passing from GPU to CPU and vice versa
        prev_crop = full_img
        # print(ranges)
        # cropping from biggest patch to smaller
        for pr in ranges:
            # print("prev_crop", prev_crop.shape)
            # print("pr in range", pr)
            # print(pr[0], pr[2], pr[1], pr[3])
            #  pr == pixel_ranges = [x0,y0,x1,y1]
            crop = prev_crop[:, :, pr[0]:pr[2], pr[1]:pr[3]]  # crop the input
            # print(crop.shape)
            crops.append(crop)
            prev_crop = crop

        # Reverse the list to compute from the fovea to the other dimensions
        #  -> I'm not sure if the computation is done in place or not, so starting from the more detailed one
        # crops = crops[::-1]
        ####
        # BEGIN Encoding
        # encoded outputs from each resolution layer
        codes = []
        for i in range(len(crops)):
            layer = crops[i]
            # print("encoding crop size = ", layer.shape)
            encs = self.encoders[i]  # encoders for the current resolution
            # downsample it as many times as needed (basically the )
            downsampler = self.downsamplers[i]  # crops are reversed => downsamplers reversed too
            layer = downsampler(layer)
            # apply all the encoders in the corresponding i'th layer
            cds = []  # codes of the corresponding layer (at the same resolution)
            for enc in encs:
                c = enc(layer)
                cds.append(c)
            codes.append(cds)
        return codes

    def decode(self, codes):
        # first decode each of the centers
        decoded_vec = []
        dec_lin_dim = []
        for i in range(len(codes)):
            cl = codes[i]
            vec = []
            for j in range(len(cl)):
                c = cl[j]
                dec = self.decoders[i][j]
                vec.append(dec(c))
            decoded_vec.append(vec)
            # decoded vector dimensions for a layer should all be the same, so taking the first one in this layer
            dec_lin_dim.append(self.decoders[i][0].conv_dim)
        # first merge each resolution independently:
        # TODO use my  TensorMergingLayer module (also to be developed before being able to use it)
        # I will do a simple mean merging instead of doing some learning, this might be good enough
        declen = len(decoded_vec)
        reslayers = []
        for i in range(declen):
            declayer = decoded_vec[i]
            rl = declayer[0]
            try:
                for l in declayer[1:]:
                    rl = rl + l
                rl = rl / len(declayer)
            except:
                # nothing happened, there was only one image in this layer
                pass
            reslayers.append(rl)

        # upsample all the layers
        upsampled = []
        for i in range(len(reslayers)):
            t = reslayers[i]
            ups = self.upsamplers[i]
            usi = ups(t)
            upsampled.append(usi)

        min_px, max_px = self._last_px_mins, self._last_px_maxs
        ranges = torch.cat([min_px, max_px], dim=1)
        # TODO reverse as the positions are defined from bigger to smaller patches but reconstruction is reversed
        # ranges = ranges[::-1]
        # for the moment just replace each higher definition patch where it belongs
        # TODO use my  TensorMergingLayer module instead (also to be developed before being able to use it)
        # print("ranges = ", ranges)
        for i in range(len(upsampled)-1):
            #  pr == pixel_ranges = [x0,y0,x1,y1]
            pr = ranges[i+1]
            bigger = upsampled[i]
            smaller = upsampled[i+1]
            # bpatch = bigger[:, :, pr[0]:pr[2], pr[1]:pr[3]]
            # print("merging ", i, pr, smaller.shape, bigger.shape, bpatch.shape)
            # print("merging ", i, smaller.shape, bigger.shape, bpatch.shape)
            bigger[:, :, pr[0]:pr[2], pr[1]:pr[3]] = smaller  # reassign the pixels to the highest definition

        # The biggest image is the one that we reconstructed
        img = upsampled[0]

        return img

    def forward(self, x, crop_centers=torch.FloatTensor([[0.5, 0.5], [0.4, 0.5], [0.3, 0.5]])):
        """
        x the input image
        crop_centers, a list of centers c where  c in [0:1],
        centers go from the larger one to the lower one
        """
        out = x
        _ = self.compute_patches(crop_centers)

        codes = self.encode(out)
        # TODO maybe ...
        # Create a simple embedding (maybe later work with a multinomial probability distribution)
        # The embeddings contain also the crop_centers, the scaling (downsample),
        #     the zoom (upsample) and the relative crop sizes to the complete image
        out = self.decode(codes)
        ###
        # I'm in doubt here if I should do the reverse process of the encoding for each encoder and then use the outputs to generate the input,
        # or I should create a single composite decoder that handles the entire reconstruction
        # First experiment -> single composite decoder ???
        # ... ???
        return out

    def save_models(self, name, path):
        raise NotImplementedError()
        pass


# NOTE: all patches will be square
# full size image will be resized to a square image, beacause it's easy

class MultiFullCAE(nn.Module):
    """
    Group of Convolutional Autoencoders for a single input resolution
    The image is treated as monochrome

    """

    def __init__(self, in_img_shape, channels=1, full_image_resize=(32, 32),
                 full_img_conv_feat=16, full_conv_sizes=[3, 5, 7, 11]):
        super(MultiFullCAE, self).__init__()
        self.channels = channels  # number of channels in the input image
        # this will be the parameter given to create the resolution encoder
        self.width = in_img_shape[0]
        self.height = in_img_shape[1]
        self.channels = channels
        self.levels = prime_factors(min(full_image_resize)).count(2)
        self.conv_sizes = full_conv_sizes  # filter sizes to create for each resolution
        self.full_image_resize = full_image_resize  # image to which to redimension the entire input image (if previous is True)
        self.full_img_conv_feat = full_img_conv_feat  # number of convolutional filters to use per layer
        self.full_conv_sizes = full_conv_sizes  # sizes of the convolutional filters, one encoder per size

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # separated as functions to be able to later LOAD the encoders instead of creating them each time
        self.downsampler = nn.AdaptiveAvgPool2d(full_image_resize)
        self.upsampler = nn.AdaptiveMaxPool2d(in_img_shape)
        self._create_encoders()
        self._create_decoders()

    def _create_encoders(self):
        for cs in self.full_conv_sizes:
            enc = CAEEncoder(self.width, self.height, self.channels, self.levels, cs, self.full_img_conv_feat)
            self.encoders.append(enc)

    def _create_decoders(self):
        for i in range(len(self.full_conv_sizes)):
            cs = self.full_conv_sizes[i]
            dec = CAEDecoder(self.encoders[i], self.width, self.height, self.channels, self.levels, cs, self.full_img_conv_feat)
            self.decoders.append(dec)

    def encode(self, x):
        # BEGIN Encoding
        # encoded outputs from each resolution layer
        # img = self.monochrome(x)
        img = self.downsampler(x)
        # apply all the encoders in the corresponding i'th layer
        codes = []
        for enc in self.encoders:
            c = enc(img)
            codes.append(c)
        return codes

    def decode(self, codes):
        # first decode each of the centers
        decoded_vec = []
        for i in range(len(codes)):
            dec = self.decoders[i]
            decoded_vec.append(dec(codes[i]))
        # TODO use my  TensorMergingLayer module (also to be developed before being able to use it)
        # I will do a simple mean merging instead of doing some learning, this might be good enough
        declen = len(decoded_vec)
        img = decoded_vec[0]
        for dr in decoded_vec[1:]:
            img = img + dr
        img = img / declen
        img = self.upsampler(img)
        return img

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out

    def save_models(self, name, path):
        raise NotImplementedError()
        pass