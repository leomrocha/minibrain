import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

from fairseq.modules.dynamic_convolution import DynamicConv1dTBC

from .utils.tools import get_activation_fn


class Conv1DBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, nlayers=5, dropout=0.1, groups=8, activation="relu"):
        """

        :param c_in: input channels
        :param c_out: output channels
        :param kernel_size:
        :param nlayers: number of convolutional layers per block
        :param dropout:
        :param groups: number of groups as in filter groups
        :param activation: activation function to use at the end of the convolutional block
        """
        super(Conv1DBlock, self).__init__()

        if c_in == c_out:
            self.use_proj = False
        else:
            self.use_proj = True

        self.convresid = weight_norm(nn.Conv1d(c_in, c_out, 1))  # [down|up]sample for residual connection if needed

        self.convs = []
        for i in range(nlayers):
            t_c_in = c_out
            if i == 0:
                t_c_in = c_in
            # Left padding
            # pad = nn.ConstantPad1d((kernel_size - 1) // 2, 0)
            cnv = weight_norm(nn.Conv1d(t_c_in, c_out, kernel_size, padding=(kernel_size-1)//2, groups=groups))
            # cnv = weight_norm(nn.Conv1d(t_c_in, c_out, kernel_size, groups=groups))
            # self.convs.append(pad)
            self.convs.append(cnv)

        self.convs = nn.Sequential(*self.convs)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        # not use padding -> is up to the main network to decide
        # res = self.leftpad(x)
        # print("1 conv1b", x.shape, x.dtype, x.is_cuda)
        res = x
        # residual connection channel dimension adaptation
        if self.use_proj:  # if in_c != out_c, need to change size of residual
            res = self.convresid(res)
        # print("2 conv1b", res.shape)
        out = self.convs(x)
        # print("3 conv1b", out.shape)
        out = self.dropout(out)
        # print("4 conv1b", out.shape)
        return self.activation(out + res)


class GatedConv1DBlock(nn.Module):
    """
    Stack of GatedConv1DBlocks
    """
    def __init__(self, c_in, c_out, kernel_size, nlayers, stride=1, dropout=0.2,
                 activation="relu", gating_activation="sigmoid"):
        """
        :param c_in: # input channels
        :param c_out: # output channels
        :param kernel_size:
        :param nlayers: number of layers to use in this stack
        :param stride:
        :param dropout:
        :param activation: if None then no activation is done
        :param gating_activation: activation layer type for the gating mechanism
        """
        super(GatedConv1DBlock, self).__init__()
        self.convs = []
        for i in range(nlayers):
            t_c_in = c_out
            if i == 0:
                t_c_in = c_in
            activ = None  # no activation for the intermediate blocks
            if i >= nlayers - 1:  # only activation for the last block
                activ = activation
            # Left padding
            # pad = nn.ConstantPad1d((kernel_size - 1) // 2, 0)
            cnv = GatedConv1DLayer(t_c_in, c_out, kernel_size, stride, dropout, activ, gating_activation)
            self.convs.append(cnv)

        self.dropout = nn.Dropout(dropout)
        self.network = nn.Sequential(*self.convs)

    def forward(self, x):
        res = x  # residual
        # TODO add positional embeddings by block
        # ret = x +
        ret = self.network(x)
        return ret + res


class GatedConv1DLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, dropout=0.2,
                 activation="relu", gating_activation="sigmoid"):
        """
        :param c_in: # input channels
        :param c_out: # output channels
        :param kernel_size:
        :param stride:
        :param dropout:
        :param activation: if None then no activation is done
        :param gating_activation: activation layer type for the gating mechanism
        """
        # TODO add Coordinate Convolution addition here (if conf) ???
        super(GatedConv1DLayer, self).__init__()
        # FIXME these values are not being taken from the input ???
        dropout = 0.2
        stride = 1
        if c_in == c_out:
            self.use_proj = 0
        else:
            self.use_proj = 1
        # Left padding left to the master network to do
        pad = (kernel_size - 1) // 2
        self.leftpad = nn.ConstantPad1d(pad, 0)
        self.convresid = weight_norm(nn.Conv1d(c_in, c_out, 1))  # downsample for residual connection if needed
        #
        self.conv1A = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=stride))
        # self.chomp1A = Chomp1d(padding)
        # gating unit
        self.conv1B = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=stride))
        # self.chomp1B = Chomp1d(padding)
        self.gatingActiv1B = nn.Sigmoid()  # get_activation_fn(gating_activation)

        # self.net1 = nn.Sequential(self.conv1A)
        self.gate1 = nn.Sequential(self.conv1B, self.gatingActiv1B)

        self.dropout1 = nn.Dropout(dropout)

        self.conv2A = weight_norm(nn.Conv1d(c_out, c_out, kernel_size, stride=stride))
        # self.chomp2A = Chomp1d(padding)
        # gating unit
        self.conv2B = weight_norm(nn.Conv1d(c_out, c_out, kernel_size, stride=stride))
        # self.chomp2B = Chomp1d(padding)
        self.gatingActiv2B = nn.Sigmoid()  # get_activation_fn(gating_activation)

        # self.activ2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net2 = nn.Sequential(self.convA, self.chomp2A)
        self.gate2 = nn.Sequential(self.conv1B, self.gatingActiv1B)

        self.activation = get_activation_fn(activation)
        # self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.convresid is not None:
    #         self.convresid.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # not use padding -> is up to the main network to decide
        # res = self.leftpad(x)
        res = x
        print(1, res.shape)
        # residual connection channel dimension adaptation
        x = self.leftpad(x)
        print(2, x.shape)
        out_net1 = self.conv1A(x)
        gt1 = self.gate1(x)
        print(3, x.shape, res.shape, out_net1.shape, gt1.shape)

        out1 = torch.mul(out_net1, gt1)
        out1 = self.dropout1(out1)
        print(4, out1.shape)

        out1 = self.leftpad(out1)
        out_net2 = self.conv2A(out1)
        print(4, out1.shape, out_net2.shape)
        gt2 = self.gate2(out1)
        print(5, out_net2.shape, gt2.shape)

        out2 = torch.mul(out_net2, gt2)
        out2 = self.dropout1(out2)
        print(6, out2.shape)

        if self.use_proj:  # if in_c != out_c, need to change size of residual
            print("convresid")
            res = self.convresid(res)
        print(1, res.shape)

        if self.activation:
            out = self.activation(out2 + res)
        else:
            out = out2 + res
        # res = x if self.downsample is None else self.downsample(x)
        # print(7, out.shape)
        return out


class DynConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, nlayers=5,
                 dropout=0.3, activation="gelu"):
        """

        :param c_in: input channels
        :param c_out: output channels
        :param kernel_size:
        :param nlayers: number of convolutional layers per block
        :param dropout:
        :param groups: number of groups as in filter groups
        :param activation: activation function to use at the end of the convolutional block
        """
        super(DynConvBlock, self).__init__()
        pass  # DynamicConv1dTBC


class ConvTimeAttSpace(nn.Module):
    """
    Convolutional block where at the end there is an Attention Module followed by a linear layer
    that reduces the number of channels. This is instead of using only Conv1D 1x1 to do this the network is more dynamic
    """
    pass