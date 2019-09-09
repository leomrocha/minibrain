
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms


# activation units allowed for the forward networks
ACTIVATIONS = {
    "relu": F.relu,
    "relu6": F.relu6,
    "sigmoid": F.sigmoid,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
    #         "linear": F.linear,  # needs extra parametrization, not used for the moment
    "logsigmoid": F.logsigmoid,
}


class FCModule(nn.Module):
    """
    Fully Connected Neural Network
    Takes as input an array with the layer input and output number of connections and the activation type of the layers
    All layers will have the same activation functions between them, except for the last one that does gives the raw output
    This is for the network is not a classifier, but intended to be used as a module somewhere and the output used in the given context
    """
    def __init__(self, layer_sizes, activation=None, p_noise=0.0, max_noise=0.0):
        """
        param: layers_sizes -> list containing the input, hidden layers and output size, minimum len = 2
                                First element is Input size
                                Last element is output size
        param: activation -> Default None, else uses the given value
        """
        super(FCModule, self).__init__()
        assert (type(layer_sizes) is tuple or type(layer_sizes) is list)
        n_layers = len(layer_sizes) - 1  # there is one layer less than the number of elements in the input
        assert (n_layers >= 1)
        self.layers = nn.ModuleList()
        self.activation = ACTIVATIONS.get(activation) or None
        for i in range(n_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.n_layers = len(self.layers)
        assert (self.n_layers == n_layers)  # should always be the same or something went really wrong

    def forward(self, x):
        # view as the layer shape
        # print(x.shape)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if self.activation is not None and i < self.n_layers - 1:
                x = self.activation(x)
        return x


class FCNet(nn.Module):
    """
    Fully Connected Neural Network Classifier, with log_softmax output
    Input Instantiation Parameters correspond to the FCModule __init__ parameters
    """
    def __init__(self, layer_sizes, activation=None, p_noise=0.0, max_noise=0.0):
        super(FCNet, self).__init__()
        self.fcnet = FCModule(layer_sizes, activation)

    def forward(self, x):
        x = self.fcnet(x)
        return F.log_softmax(x, dim=1)
        # return F.log_softmax(x)


class ColumnNet(nn.Module):
    """
    Neural Network that contains several Fully Connected Neural Networks, each having a set number of layers
    Takes as input an array with the layer input and output number of connections and the activation type of the layers
    All layers will have the same activation functions between them, except for the last one that does gives the raw output
    This is for the network is not a classifier, but intended to be used as a module somewhere and the output used in the given context
    """
    def __init__(self, layer_sizes, activations, last_layer=10, p_noise=0.0, max_noise=0.0):
        """

        :param layer_sizes: bidimensional array
        :param last_layer:
        :param activation:
        """
        super(ColumnNet, self).__init__()
        assert(len(layer_sizes) == len(activations))
        self.nets = nn.ModuleList()
        self.catlayer_size = sum([l[-1] for l in layer_sizes])  # get the last layers embedding sizes of each subnetwork
        for inet, activation in zip(layer_sizes, activations):
            self.nets.append(FCModule(inet, activation))
        self.last = nn.Linear(self.catlayer_size, last_layer)

    def forward(self, x, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        xis = []
        for n in self.nets:
            xis.append(n(x))
        xcat = torch.cat(xis, dim=1)  # .to(device)
        x = F.relu(self.last(xcat))
        return F.log_softmax(x, dim=1)
        # return F.log_softmax(x)


# Note: I could add the sparsity to the previous FCNet but I want to make it explicit here
# something is not working on this one, NLLLoss diverges
class SparseNet(nn.Module):
    """
    Sparse Forward Neural Network, sparsity level is provided as a probability of elements being set to 0

    """
    def __init__(self, layer_sizes, sparsity=0.4, activation=None, name="SparseNet",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 p_noise=0.0, max_noise=0.0):
        """
        param: layers_sizes -> list containing the input, hidden layers and output size, minimum len = 2
                                First element is Input size
                                Last element is output size
        param: sparsity -> percentage of an element being non zero
        param: activation -> Default None, else uses the given value
        """
        super(SparseNet, self).__init__()
        assert (type(layer_sizes) is tuple or type(layer_sizes) is list)
        n_layers = len(layer_sizes) - 1  # there is one layer less than the number of elements in the input
        assert (n_layers >= 1)
        self.name = name
        self.device = device
        self.layers = nn.ModuleList()
        self.sparsity = 1 - sparsity
        self.masks = []
        self.activation = ACTIVATIONS.get(activation) or None
        for i in range(n_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            t_mask = torch.FloatTensor(layer_sizes[i + 1]).uniform_() > (1 - sparsity)
            self.masks.append(t_mask.to(device).float())

        self.n_layers = len(self.layers)
        assert (self.n_layers == n_layers)  # should always be the same or something went really wrong

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if self.activation is not None and i < self.n_layers - 1:
                x = self.activation(x)
                x = x * self.masks[i]
        return x


class ConvNet(nn.Module):
    """
    Convolutional Neural Network
    TODO Can add spatial coordinate features ( ... TODO check which was the paper that introduced that)
    TODO Can add frequency domain transforms (Fourier)
    """

    def __init__(self, width, height, channels=1, kernel_sizes=[3, 3, 3],
                 n_features=[32, 64, 48], p_noise=0.0, max_noise=0.0):
        super(ConvNet, self).__init__()
        # print("ConvNet init ", width, height, channels,  kernel_size, n_features)
        self.width = width
        self.height = height
        self.channels = channels
        # compute the maximum number of levels that this resolution can handle,
        # this will be the parameter given to create the resolution encoder
        assert(len(kernel_sizes) == len(n_features))
        self.levels = len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.n_features = n_features
        self.paddings = [k // 2 for k in kernel_sizes]
        self.indices = []

        self.l_features = [channels]
        self.layers = nn.ModuleList()
    # TODO
    #     for i in range(self.levels + 1):
    #         self.l_features.append(first_feature_count * (2 ** (i)))
    #
    #     for i in range(self.levels):
    #         nfeat = self.l_features[i + 1]
    #         layer = nn.Sequential(
    #             nn.Conv2d(self.l_features[i], nfeat, kernel_size=kernel_size, padding=padding),
    #             nn.ReLU(),
    #             nn.Conv2d(nfeat, nfeat, kernel_size=kernel_size, padding=padding),
    #             nn.ReLU(),
    #             torch.nn.MaxPool2d(2, stride=2, return_indices=True)
    #         )
    #         self.layers.append(layer)
    #
    #     self.conv_dim = ((width * height) // ((2 ** levels) ** 2)) * self.l_features[-1]
    #
    # def forward(self, x):
    #     self.indices = []
    #     out = x
    #     for i in range(self.levels):
    #         layer = self.layers[i]
    #         out, idx = layer(out)
    #         self.indices.append(idx)
    #     return out

