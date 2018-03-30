import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision import datasets
from torchvision.utils import save_image

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

        # self.conv_dim = int(((w*h)/ ((2**levels)**2)) * self.l_features[-1])
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
        self.layers = self.layers[::-1]

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


#definitions of the operations for the full image autoencoder
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406], # from example here https://github.com/pytorch/examples/blob/409a7262dcfa7906a92aeac25ee7d413baa88b67/imagenet/main.py#L94-L95
   std=[0.229, 0.224, 0.225]
#   mean=[0.5, 0.5, 0.5], # from example here http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#    std=[0.5, 0.5, 0.5]
)

#the whole image gets resized to a small image that can be quickly analyzed to get important points
def fullimage_preprocess(w=48,h=48):
    return transforms.Compose([
        transforms.Resize((w,h)), #this should be used ONLY if the image is bigger than this size
        transforms.ToTensor(),
        normalize
    ])

#the full resolution fovea just is a small 12x12 patch 
full_resolution_crop = transforms.Compose([
    transforms.RandomCrop(12),
    transforms.ToTensor(),
    normalize
    ])

def downsampleTensor(crop_size, final_size=16):
    sample = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.Resize(final_size), 
        transforms.ToTensor(),
        normalize
    ])
    return sample


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


# Hyper Parameters
# num_epochs = 5
# batch_size = 100
# learning_rate = 0.001


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 12, 12)
    return x

def main():

    num_epochs = 100
    batch_size = 128
    learning_rate = 0.0001
    #learning_rate = 0.001

    model = CAE(12, 12, 3, 2, 128).cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    transformation = full_resolution_crop
    train_loader, test_loader = get_loaders(batch_size, transformation)


    for epoch in range(num_epochs):
        for i, (img, labels) in enumerate(train_loader):
            img = Variable(img).cuda()
            # ===================forward=====================
    #         print("encoding batch of  images")
            output = model(img)
    #         print("computing loss")
            loss = criterion(output, img)
            # ===================backward====================
    #         print("Backward ")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            in_pic = to_img(img.cpu().data)
            save_image(pic, './cae_results/2x2-out_image_{}.png'.format(epoch))
            save_image(in_pic, './cae_results/2x2-in_image_{}.png'.format(epoch))
        if loss.data[0] < 0.15: #arbitrary number because I saw that it works well enough
            break


    model.save_model("2x2-layer", "CAE")
    
if __name__ == "__main__":
    main()
