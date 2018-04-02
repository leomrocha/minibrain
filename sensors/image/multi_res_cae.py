

import torch
from torch import nn, optim
# import torchvision
# from torch.nn import functional as F
# from torch.autograd import Variable
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms, utils
# from torchvision import datasets
# from torchvision.utils import save_image

from .cae import *
from .helpers import *
from .helper_modules import *


# NOTE: all patches will be square
# full size image will be resized to a square image, beacause it's easy

class MultiResCAEEncoder(nn.Module):
    """
    Multi Resolution group of Convolutional Autoencoders
    This module intends to group several autoencoders that accompany different resolutions
    The goal of this module is be able to train and maintain all the filters in one place
    This model can be saved and loaded as a single element
    the FULL IMAGE is not considered in this, only the foveated multi resolution
    """
    # TODO This model outputs tensor of dimension 1x1xN that is the concatenation of the output of all the encoders ensemble

    def __init__(self, in_img_shape, channels=3, res_levels=3, conv_layer_feat=[32, 16, 16],
                 res_px=[[12, 12], [16, 16], [20, 20]], crop_sizes=[[12, 12], [32, 32], [64, 64]],
                 # conv_sizes = [(3,5,7), (3,5,7,11), (3,5,7,11)],  # this is too much I think
                 conv_sizes=[(3, 5, 7), (3, 5), (3, 5)]):
        """
        @param in_imag_shape : [width, height]  # the input image shape, to be able to pre-compute the transform matrices
        """
        super(CAE, self).__init__()
        self.channels = channels  # number of channels in the input image
        self.res_levels = res_levels  # number of resolution levels (NOT including the full image)
        self.conv_layer_feat = conv_layer_feat  # number of convolutional filters per CAE in the first level
        self.res_px = res_px  # downsampled resolution in pixels for each resolution
        self.conv_sizes = conv_sizes  # conv filter sizes per layer, one encoder per size per layer
        ##
        # compute the maximum number of levels that this resolution can handle,
        # this will be the parameter given to create the resolution encoder
        self.max_levels = [prime_factors[min(i)].count(2) for i in res_px]
        ##
        # Pre-computing cropping matrices
        self.crop_sizes = torch.IntTensor(crop_sizes)  # Ps - Patches sizes -  size of the patch to crop
        self.half_crop_sizes = torch.IntTensor(
            crop_sizes)  # Ps/2 - Patches half sizes -  half size of the patch to crop, to compute positions
        self.ref_patch = torch.IntTensor(conv_sizes[:].append(in_img_shape)[::-1])  # RP - Reference Patches
        # pre-compute Patch Dynamic Range (pixel wise)
        self._pdr = self.ref_patch - self.crop_sizes
        # saves the last patch centers
        self._last_centers = None  # make a variable placeholder here
        # saves the last patch ranges
        self._last_px_mins = None
        self._last_px_maxs = None
        ##
        # Actual work is done in the following modules
        #
        self.downsamplers = [DownsamplerLayer(w, h) for w, h in res_px]
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

            for j in range(l_conv_sizes):
                enc = CAEEncoder(w, h, c, levels, j, conv_features)
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

            for j in range(l_conv_sizes):
                enc = CAEDecoder(self.encoders[i][j], w, h, c, levels, j, conv_features)
                res_decoders.append(enc)
            self.decoders.append(res_decoders)

    def compute_patches(self, crop_centers):
        """
        Computes the ranges that have to be cropped from the input image.
        The computation follows the principles:
         - The inner patches positions are relative to the outer ones, in the corresponding hierarchy
         - The patches can not go off the container image

        @param crop_centers MUST be a HalfTensor or FloatTensor
        @returns : (centers, min_points, max_points) two IntTensors with the center and ranges that each patch occupies

        The returned elements contain the patches from the bigger to the smaller one
        """
        # formula follows this -> see how to implement it in vector operations to make it faster, or precompute it for a fixed image size
        # x_min = 0 + patch_width/2
        # x_max = full_img.width - patch_width/2
        # x_pos = c * (x_max - x_min) == full_img.width - patch_width
        # patch = img[x_pos - patch_width/2 : x_pos + patch_width/2] (warning on patch size)

        self._last_centers = crop_centers * self._pdr  # element wise multiplication
        self._last_px_mins = self._last_centers - self.half_crop_sizes
        self._last_px_maxs = self._last_centers + self.half_crop_sizes

        return self._last_centers, self._last_px_mins, self._last_px_maxs

    def encode(self, x, crop_centers):
        full_img = x
        ########################
        # BEGIN Cropping
        ###
        # compute the patches positions
        centers, min_px, max_px = self.compute_patches(crop_centers)
        # make only one vector out of the needed ones
        ranges = min_px.cat(max_px)
        # Get all the cropped layers
        crops = []
        # TODO find out how to do this without passing from GPU to CPU and vice versa
        prev_crop = full_img
        for pr in ranges:
            #  pr == pixel_ranges = [x0,y0,x1,y1]
            crop = prev_crop[pr[0]:pr[2], pr[1]:pr[3]]  # crop the input
            crops.append(crop)
            prev_crop = crop

        # Reverse the list to compute from the fovea to the other dimensions -> I'm not sure if the computation is done in place or not, so starting from the more detailed one
        crops = crops[::-1]
        # encoded outputs from each resolution layer
        codes = []
        for i in range(len(crops)):
            layer = crops(i)
            encs = self.encoders[i]  # encoders for the current resolution
            # downsample it as many times as needed (basically the )
            downsampler = self.downsamplers[i]
            layer = downsampler(layer)
            # apply all the encoders in the corresponding i'th layer
            cds = []  # codes of the corresponding layer (at the same resolution)
            for enc in encs:
                c = enc(layer)
                cds.append(c)
            codes.append(cds)
        return codes

    def decode(self, codes, crop_centers):
        # first decode each of the centers
        decoded_vec = []
        for i in range(len(codes)):
            cl = codes[i]
            vec = []
            for j in range(len(cl)):
                c = cl[j]
                dec = self.decoders[i][j]
                decoded_vec.append(dec(c))
        # first merge each resolution independently:
        # I will do a simple mean merging instead of doing some learning, this might be good enough
        # TODO use my  TensorMergingLayer module (also to be developed before being able to use it)
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

        # and now recreate the image using also the location information
        # this is the difficult part
        # I might want here to work on the Overcomplete Partitioned Dendritic Layers before going on ... seems hard here
        # using a fully connected layer might be a bit slow to train and will introduce too much noise as there are
        # non desired inputs
        # TODO ... first need to create and test the Overcomplete Partitioned Dendritic Layers
        img = None

        return img

    def forward(self, x, crop_centers=torch.HalfTensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])):
        """
        x the input image
        crop_centers, a list of centers c where  c in [0:1],
        centers go from the larger one to the lower one
        """
        out = x
        codes = self.encode(x, crop_centers)
        # TODO maybe ...
        # Create a simple embedding (maybe later work with a multinomial probability distribution)
        # The embeddings contain also the crop_centers, the scaling (downsample),
        #     the zoom (upsample) and the relative crop sizes to the complete image
        out = self.decode(codes, crop_centers)
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

    def __init__(self, channels=1, ds_full_image_cae=True, full_image_size=32, full_img_conv_feat=16,
                 full_conv_sizes=(3, 5, 7)):
        super(CAE, self).__init__()
        self.channels = channels  # number of channels in the input image
        # this will be the parameter given to create the resolution encoder
        self.levels = prime_factors[full_image_size].count(2)
        self.conv_sizes = full_conv_sizes  # filter sizes to create for each resolution
        self.ds_full_img_cae = ds_full_image_cae  # indicate if create or not the full image downsample conv encoder
        self.full_image_size = full_image_size  # image to which to redimension the entire input image (if previous is True)
        self.full_img_conv_feat = full_img_conv_feat  # number of convolutional filters to use per layer
        self.full_conv_sizes = full_conv_sizes  # sizes of the convolutional filters, one encoder per size

        self.full_encoders = nn.ModuleList()
        self.full_decoders = nn.ModuleList()
        # separated as functions to be able to later LOAD the encoders instead of creating them each time
        self._create_full_encoders()
        self._create_full_decoders()

    def _create_full_encoders(self, channels=1):
        for cs in self.full_conv_sizes:
            width = height = self.full_image_size
            channels = self.channels  # although I'm thinking in making this monochrome instead to save processing time
            enc = CAEEncoder(width, height, channels, self.levels, cs, self.full_img_conv_feat)
            self.full_encoders.append(enc)

    def _create_full_decoders(self, channels=1):
        for i in range(self.full_conv_sizes):
            cs = self.full_conv_sizes[i]
            width = height = self.full_image_size
            channels = self.channels  # although I'm thinking in making this monochrome instead to save processing time
            enc = CAEDecoder(self.full_encoders[i], width, height, channels, self.levels, cs, self.full_img_conv_feat)
            self.full_decoders.append(enc)

    def forward(self, x):
        out = x
        # input = downsampled full image converted to monochrome
        ########################
        # BEGIN Encoding
        ###

        # for the moment this full image is computed each time, but in the future this will be
        #     done ONLY if the input image changes
        #     maybe what we want to work with is only the difference from previous frames -> future when working in dynamic environments
        # encoder full downsampled image
        #
        # join  all encodings into a single vector
        # END Encoding
        ########################
        # BEGIN decoding
        return out

    def save_models(self, name, path):
        raise NotImplementedError()
        pass