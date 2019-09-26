"""
Module that intends to do some trials on guessing the next character prediction,
I want to see if this works OK with the multihot encoding of utf-8.

I would like to test it on several languages and cover all the character spectrum but it seems that getting a nice DB
is difficult, so I'll start with one or two languages, I'll start with wikitext-2-raw

The network uses the idea of multi resolution and instead of using bidirectional LSTM like FLAIR
Contextual String Embeddings for Sequence Labeling - Alan Akbik Duncan Blythe Roland Vollgraf

but unlike it will be using convolutional neural networks for the embeddings and explicit memories will be used
in later higher hierarchical stage to keep track of higher level concepts


Levels:

First there is the utf-8 encoder layer

second there is a convolutional neural network with "memory" of N elements

A TCN (Temporal Convolutional Neural Network) is used there

This goes to a FC layer with a variational auto encoder like latent embedding
(later I'd like to use a more complex latent space ... maybe a complex space?)

This latent space should represent the current context from which the next character should be decoded


Next steps is using attention models with example taking

Hierarchical Neural Story Generation - Angela Fan, Mike Lewis, Yann Dauphin
https://arxiv.org/abs/1805.04833

"""

from torch import nn, tensor, optim
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from tcn import TemporalConvNet


class CharConvPredictor(nn.Module):
    """
    UTF-8 Encoder for next character prediction
    """
    def __init__(self, fc_hid_dim, fc_out_dim,
                 num_inputs, num_channels, kernel_size=2, dropout=0.2
                 ):
        """

        """
        super(CharConvPredictor, self).__init__()
        # encoder layer ... should use maybe F.embedding with the utf-8 matrix

        # TCN
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        # variational latent space ... wait a minute ... here I didn't think about the temporal part on the FC layers
        self.fc21 = nn.Linear(fc_hid_dim, fc_out_dim)
        self.fc22 = nn.Linear(fc_hid_dim, fc_out_dim)
        # output
        # pass

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = tensor(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # encode

        #
        x = self.fc1(x)
        # x = self.fc2(x)
        x = F.relu(x)  # or better use another kind of activation function?
        mu, logvar = self.fc21(x), self.fc22(x)
        return self.reparameterize(mu, logvar), mu, logvar
