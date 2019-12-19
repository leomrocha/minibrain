import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from models.pt_model import PyTorchModel


class PlasticLayer(nn.Module):
    """
    PlasticLayer implements hebbian basic and OJA's rules for learning in a fully connected manner.
    This class is mostly based on the differentiable-plasticity work from Uber here:
    https://github.com/uber-research/differentiable-plasticity hence the same license as that repo is applied to this file
    """
    def __init__(self, in_size, out_size, plasticity="hebbian", device="cpu"):
        """
        :param embedding_size: size of the layer in neurons
        :param plasticity: [nonplastic|hebbian|oja]
        """
        super(PlasticLayer, self).__init__()
        # TODO fix this limit on input and output, this is temporal while I do the tests
        # this is because is easier for the moment to track square matrices instead
        assert(in_size == out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.embedding_size = embedding_size = in_size
        self.plasticity = plasticity
        self.device = device
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order,
        #  following apparent pytorch conventions
        # Each *column* of w targets a single output neuron
        # The matrix of fixed (baseline) weights
        self.w = tensor(.01 * torch.randn(embedding_size, embedding_size), requires_grad=True).to(device)
        # The matrix of plasticity coefficients
        self.alpha = tensor(.01 * torch.randn(embedding_size, embedding_size), requires_grad=True).to(device)
        # The eta coefficient is learned
        self.eta = tensor(.01 * torch.ones(1), requires_grad=True).to(device)
        if self.plasticity == 'nonplastic':
            self.zero_diag_alpha()  # No plastic autapses
        elif self.plasticity == "hebbian":
            pass
        elif self.plasticity == "oja":
            pass
        else:
            raise ValueError("Wrong network type!")

    def forward(self, input, yin, hebb):
        # Run the network for one timestep
        # print(self.plasticity)
        if self.plasticity == 'hebbian':
            yout = F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb)) + input)
            # bmm used to implement outer product with the help of unsqueeze (i.e. added empty dimensions)
            hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0]
        elif self.plasticity == 'oja':
            # yout = F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb)) + input)
            # # Oja's rule.  TODO check that this is OK in the dimensions ... I might have messed something
            # hebb = hebb + self.eta * torch.mul((yin.unsqueeze(2) - torch.mul(hebb, yout.unsqueeze(1))),
            #                                    yout.unsqueeze(1))[0]
            raise NotImplementedError("TODO check this")
        elif self.plasticity == 'nonplastic':
            yout = F.tanh(yin.mm(self.w) + input)
        else:
            raise ValueError("Wrong network type in forward()!")
        return yout, hebb

    def initial_zero_state(self):
        return tensor(torch.zeros(1, self.embedding_size)).to(self.device)

    def initial_zero_hebb(self):
        return tensor(torch.zeros(self.embedding_size, self.embedding_size)).to(self.device)

    def zero_diag_alpha(self):
        # Zero out the diagonal of the matrix of alpha coefficients: no plastic autapses
        self.alpha.data -= torch.diag(torch.diag(self.alpha.data)).to(self.device)


class PlasticFCNetwork(nn.Module):
    def __init__(self, in_size, out_size, time_steps, num_channels, plasticity="hebbian", device="cpu"):
        """
        :param embedding_size: size of the layer in neurons
        :param plasticity: [nonplastic|hebbian|oja]
        """
        super(PlasticFCNetwork, self).__init__()
        # TODO
        self.input_size = in_size
        self.output_size = out_size
        self.time_steps = time_steps
        self.num_channels = num_channels
        self.plasticity = plasticity
        self.device = device

        self.hebbian_layers = []
        self.embeddings = nn.Embedding(in_size, num_channels[0])
        for i in range(1, len(num_channels)):
            in_channels = num_channels[i-1]
            out_channels = num_channels[i]
            self.hebbian_layers.append(PlasticLayer(in_channels, out_channels, plasticity, device))
        # self.relu = nn.ReLu()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

        #internal memory state
        self._hebb = [hb.initial_zero_hebb() for hb in self.hebbian_layers]
        self._y = []
        self._x = []

    def init_weights(self):
        self._hebb = [hb.initial_zero_hebb() for hb in self.hebbian_layers]

    def forward(self, x):
        # TODO here is where I have to deal with the iterations
        # print(x.shape)
        # raise NotImplementedError("TODO")
        embd = self.embeddings(x)
        #####
        device = self.device
        # first implementation, basic loop through everything
        # TODO check here as this might be the issue with Episodes learning!!!
        # (also) the loop is NOT GOOD for learning speed
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size)).to(device)
        for s in range(x.shape[0]):  # sample
            for t in range(x.shape[1]):  # time
                self._x = [embd[s][t]] + [torch.zeros_like(embd[s][t]).to(device)] * len(self.hebbian_layers)
                self._y = [torch.zeros_like(self._x[0]).to(device)] * len(self.hebbian_layers)
                for i_hb in range(len(self.hebbian_layers)):
                    hb = self.hebbian_layers[i_hb]
                    y, hebb = hb(self._x[i_hb], self._y[i_hb], self._hebb[i_hb])
                    self._x[i_hb + 1] = y
                    self._y[i_hb] = y
                    self._hebb[i_hb] = hebb
                #sigmoid of the last hebbian layer
                o = self.sig(self._x[-1])
                out[s][t] = self.softmax(o)
        return out
        #####
        # test n# 2
        # self._x = [embd] + [torch.zeros_like(embd).to(device)] * len(self.hebbian_layers)
        # self._y = [torch.zeros_like(self._x[0]).to(device)] * len(self.hebbian_layers)
        # for i_hb in range(len(self.hebbian_layers)):
        #     hb = self.hebbian_layers[i_hb]
        #     y, hebb = hb(self._x[i_hb], self._y[i_hb], self._hebb[i_hb])
        #     self._x[i_hb + 1] = y
        #     self._y[i_hb] = y
        #     self._hebb[i_hb] = hebb
        # #sigmoid of the last hebbian layer
        # out = self.softmax(self.sig(self._x[-1]))
        return out


class PlasticFCBaseline(PyTorchModel):
    """
    TCNBaseline
    """
    def __init__(self, config):
        super(PlasticFCBaseline, self).__init__(config)
        # self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._hidden_size = self._config['hidden_size']
        self._rule = self._config['hebbian_rule']
        # print(self._rule)
        self._time_steps = self._config['max_len']
        # self._num_channels = [self._input_size, self._config["embedding_size"], self._config["embedding_size"]]
        self._num_channels = [self._embd_size] + [self._hidden_size] * self._n_layers

        self.model = PlasticFCNetwork(in_size=self._input_size, out_size=self._input_size, time_steps=self._time_steps,
                                      num_channels=self._num_channels, plasticity=self._rule, device=self.device)

        self.model.to(self.device)
