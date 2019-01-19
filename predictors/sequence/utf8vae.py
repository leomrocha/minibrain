
import torch
# import torchvision
from torch import nn, tensor
from torch.nn import functional as F
# from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

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

        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)  # later try without this extra layer ... but I want non linearities
        self.fc21 = nn.Linear(hid_dim, out_dim)
        self.fc22 = nn.Linear(hid_dim, out_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = tensor(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu    
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu, logvar = self.fc21(x), self.fc22(x)
        return self.reparameterize(mu, logvar), mu, logvar


class UTF8Decoder(torch.nn.Module):
    """
    The Decoder = P(X|z) for the Network
    As a Variational AutoEncoder with internal linear units

    In particular this decoder is done so it can decode each segment part of the utf-8 multihot encoding.
    This is done in the following way:
      First a linear layer adapts the input size to the size of the UTF-8 code, this is from a defined UTF-8 table
      Then for each part we do a decoding column that will decode the segment
          4 elements that indicate the number of segments (the segment part) used
          2^8 (8 bits in one hot) elements that indicate the first segment
          2^6 (6 bits in one hot) elements that indicate the second segment
          2^6 elements that indicate the third segment
          2^6 elements that indicate the fourth segment
      Then each part is decoded with LogSoftmax and these are concatenated
    """

    def __init__(self, in_dim, segments=4):
        super(UTF8Decoder, self).__init__()
        assert(0 < segments <= 4)

        # TODO these numbers should be in a constants package
        self.code_dim = 4 + (2 ** 8) + (segments - 1) * (2 ** 6)
        # self.code = torch.zeros(self.code_dim)
        self.dimensions = [4, 2**8, 2**6, 2**6, 2**6][:segments+1]

        # this layer adapts the input layer size to the size of the utf-8 code
        self.in_linear = nn.Linear(in_dim, self.code_dim)

        # now we have something that we can use to start encoding
        self.decoders = []
        for d in range(self.dimensions):
            dec = nn.Sequential(
                nn.Linear(self.code_dim, d),
                # torch.nn.Linear(2*d, d),  # later maybe try with this extra layer ... but might not add too much
                nn.ReLU(True),  # inplace
                nn.LogSoftmax(dec)
            )
            self.decoders.append(dec)

    def forward(self, x):
        x = F.relu(self.in_linear(x))
        res = [decoder(x) for decoder in self.decoders]
        out = torch.cat(res, dim=0)  # TODO check if this is the right dimension to concatenate
        return out

    
class UTF8VAE(nn.Module):
    """
    Variational Auto Encoder
    """
    def __init__(self, in_code_dim, hid_dim, code_dim=100, segments=4):
        super(UTF8VAE, self).__init__()
        self.hid_dim = hid_dim
        self.in_code_dim = in_code_dim
        self.segments = segments
        self.encoder = VAEEncoder(in_code_dim, hid_dim, code_dim)
        self.decoder = UTF8Decoder(code_dim, segments=segments)
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x.view(-1, self.in_code_dim))
        return self.decoder(z), mu, logvar
        
    def save_model(self, name, path):
        torch.save(self.encoder, os.path.join(path, "utf8_vae_encoder_"+name+"_seg-"+self.segments+".pth"))
        torch.save(self.decoder, os.path.join(path, "utf8_vae_decoder_"+name+"_seg-"+self.segments+".pth"))


# def get_loaders(batch_size, transformation, dataset = datasets.CIFAR100, cuda=True):
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         dataset('../data', train=True, download=True,
#                        transform=transformation),
#         batch_size=batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         dataset('../data', train=False, transform=transformation),
#         batch_size=batch_size, shuffle=True, **kwargs)
#
#     return train_loader, test_loader
#
#
# def train(model, optimizer, loss_function, train_loader, epoch, batch_size, width, height, channels, log_interval=100, cuda=True):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = Variable(data)
#         if cuda:
#             data = data.cuda()
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar, width, height, channels)
#         loss.backward()
#         train_loss += loss.data[0]
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))
#
#
# def test(name, model, test_loader, epoch, batch_size, width, height, channels, cuda=True):
#     model.eval()
#     test_loss = 0
#     for i, (data, _) in enumerate(test_loader):
#         data = Variable(data, volatile=True)
#         if cuda:
#             data = data.cuda()
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar, width, height, channels).data[0]
#         if i == 0:
#             n = min(data.size(0), 100)
#             comparison = torch.cat([data[:n],
#                                   recon_batch.view(batch_size, channels, width, height)[:n]])
#             save_image(comparison.data.cpu(),'vae_results/' + name + '_reconstruction_' + str(epoch) + '.png', nrow=n)
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
#
#
# def train_save_model(name, model, optimizer, loss_function, transformation, width, height, channels, epochs=10, batch_size=128, dataset = datasets.CIFAR100, cuda=True):
#
#
#     train_loader, test_loader = get_loaders(batch_size, transformation)
#
#     for epoch in range(1, epochs + 1):
#         train(model, optimizer, loss_function, train_loader,epoch, batch_size, width, height, channels)
#         test(name, model, test_loader,epoch, batch_size, width, height, channels)
#
#     #torch.save(model, name+".pth")
#     model.save_model(name, "VAE")
#
#
# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar, width, height, channels):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, width*height*channels), size_average=False)
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return BCE + KLD
#
#
# def train_all(models, epochs=10):
#     for n,m,t in models:
#         print("--------------------------------------------------")
#         print("training model "+ n)
#         m.cuda()
#         optimizer = optim.Adam(m.parameters(), lr=1e-3)
#         train_save_model(n, m, optimizer, loss_function, t, m.width, m.height, m.channels, dataset=datasets.CocoDetection, epochs=epochs)
#
#
# if __name__ == "__main__":
#     #numbers are chosen so the image size allows for at least 2 convolution layers in a CNN (they'll be used for other parts)
#     models = [
#           ("fovea_12x12", VAE(12,12,3,500,100),full_resolution_crop),
#           ("downsample1_16x16-30x30", VAE(16,16,3,500,100), downsampleTensor(30,16)),
#           # commented due to image size, it seems that some images fail the training, I'll need to find a good dataset
#           # ("downsample2_16x16-60x60", VAE(16,16,3,500,100), downsampleTensor(60,16)),
#           # ("downsample3_16x16-120x120", VAE(16,16,3,500,100), downsampleTensor(120,16)),
#           # ("downsample4_20x20-240x240", VAE(20,20,3,500,100), downsampleTensor(240,20)),
#           # ("downsample5_20x20-480x480", VAE(20,20,3,500,100), downsampleTensor(480,20)),
#           ("fullimage_20", VAE(20,20,3,500,100), fullimage_preprocess(20,20)),
#           ("fullimage_48", VAE(48,48,3,500,100), fullimage_preprocess(48,48)),
#           ("fullimage_72", VAE(72,72,3,500,100), fullimage_preprocess(72,72)),
#           ("fullimage_100", VAE(96,96,3,500,100), fullimage_preprocess(96,96)),
#          ]
#     train_all(models)