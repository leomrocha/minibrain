import torch
from torch import nn, tensor, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
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
        """
        Variational Auto Encoder
        :param in_dim: input dimension, for the case of UTF8 encoding will be depending on the number of segments
        of the encoding
        :param hid_dim: hidden encoder dimension
        :param out_dim: dimension of the encoding
        """
        super(VAEEncoder, self).__init__()

        self.fc1 = nn.Linear(in_dim, hid_dim)
        # self.fc2 = nn.Linear(hid_dim, hid_dim)  # later try without this extra layer ... but I want non linearities
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
        # x = self.fc2(x)
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
        for d in self.dimensions:
            dec = nn.Sequential(
                nn.Linear(self.code_dim, d),
                # torch.nn.Linear(2*d, d),  # later maybe try with this extra layer ... but might not add too much
                nn.ReLU(True),  # inplace
                nn.LogSoftmax()
            )
            self.decoders.append(dec)

    def forward(self, x):
        x = F.relu(self.in_linear(x))
        res = [decoder(x) for decoder in self.decoders]
        # print("results shapes:")
        # for r in res:
            # print(r.shape)
        out = torch.cat(res, dim=1)  # TODO check if this is the right dimension to concatenate
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


def train(model, optimizer, loss_function, train_loader, epoch, vector_size, channels, log_interval=100, cuda=True):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = tensor(data)
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, vector_size, channels)
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


def test(model, test_loader, epoch, vector_size, channels, cuda=True):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = tensor(data, volatile=True)
        if cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, vector_size, channels).data[0]

    test_loss /= len(test_loader.dataset)
    print('epoch: {}====> Test set loss: {:.4f}'.format(epoch, test_loss))

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


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, vector_size, channels=1):
    # print("x shape = ", x.shape, recon_x.shape)
    # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.nll_loss(recon_x, x)
    BCE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


class UTF8Dataset(Dataset):
    def __init__(self, datafile, device="gpu"):
        self.codedata = torch.from_numpy(np.load(datafile))  # .to(device)

    def __getitem__(self, index):
        return self.codedata[index]

    def __len__(self):
        return len(self)


def train_overfit():
    # generate dataset inputs (basically the same as the encoding)
    # We are going to overfit
    epochs = 100
    segments = 3  # I do with 3 as it will be much MUCH faster and smaller for my resources than 4 segments
    in_size = 388  # 3 segments
    hidd_size = 100
    code_size = 50
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vector_size = code_size
    channels = 1
    batch_size = 10
    datafile = "utf8_code_matrix_3seg.npy"
    log_interval = 10

    model = UTF8VAE(in_size, hidd_size, code_size, segments=segments)
    # loader = DataLoader(UTF8Dataset("utf8_code_matrix_3seg.npy"), batch_size=batch_size)

    name = "utf8-vae-3segments-overfit"
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_batches = 10
    n_epocs = 100
    # train_loader, test_loader = get_loaders(batch_size, transformation)
    # we are overfitting, so train and test is the same thing.

    #     for epoch in range(1, epochs + 1):
    #         train(model, optimizer, loss_function, loader,epoch, vector_size, channels)
    #         test(model, loader, epoch, vector_size, channels)
    data = torch.from_numpy(np.load(datafile)).float()
    data = data  # .to(device)
    model = model  # .to(device)

    model.train()
    for epoch in range(n_epocs):
        train_loss = 0
        for batch_idx in range(n_batches):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, vector_size, channels)
            loss.backward()
            train_loss += loss.data.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data),
                           100. * batch_idx / len(data),
                           loss.data.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(data)))

    model.save_model(name, "saved_models")
    # train_all(models)


if __name__ == "__main__":
    train_overfit()
