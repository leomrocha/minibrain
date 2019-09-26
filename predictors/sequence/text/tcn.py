# From https://github.com/locuslab/TCN
# License for this file in the original repository, MIT at the moment of the writing of this note
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Conv1DAttentionBlock(nn.Module):
    """
    Attention implemented with convolutions instead of
    """
    pass


class GatedConv1DBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, dropout=0.2, activation="relu"):
        """
        :param c_in: # input channels
        :param c_out: # output channels
        :param kernel_size:
        :param stride:
        :param dropout:
        :param activation: if None then no activation is done
        """
        # TODO add Coordinate Convolution addition here (if conf)
        super(GatedConv1DBlock, self).__init__()

        if c_in == c_out:
            self.use_proj = 0
        else:
            self.use_proj = 1
        # Left padding left to the master network to do
        # self.leftpad = nn.ConstantPad1d(kernel_size - 1)
        self.convresid = weight_norm(nn.Conv1d(c_in, c_out, 1))  # downsample for residual connection if needed
        #
        self.conv1A = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=stride))
        # self.chomp1A = Chomp1d(padding)
        # gating unit
        self.conv1B = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=stride))
        # self.chomp1B = Chomp1d(padding)
        self.gatingActiv1B = nn.Sigmoid()  # nn.Tanh()  #

        # self.net1 = nn.Sequential(self.conv1A)
        self.gate1 = nn.Sequential(self.conv1B, self.gatingActiv1B)

        # self.activ1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2A = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=stride))
        # self.chomp2A = Chomp1d(padding)
        # gating unit
        self.conv2B = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=stride))
        # self.chomp2B = Chomp1d(padding)
        self.gatingActiv2B = nn.Sigmoid()  # nn.Tanh()  #

        # self.activ2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net2 = nn.Sequential(self.convA, self.chomp2A)
        self.gate2 = nn.Sequential(self.convB, self.gatingActivB)

        # TODO improve the activation selection mechanism (take out the ifs and make it more meta)
        self.activation = None
        if activation and activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation and activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation and activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        # self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # not use padding -> is up to the main network to decide
        # res = self.leftpad(x)
        res = x

        # residual connection channel dimension adaptation
        if self.use_proj == 1:  # if in_c != out_c, need to change size of residual
            res = self.convresid(res)

        out_net1 = self.conv1A(res)
        gt1 = self.gate1(res)

        out1 = torch.mul(out_net1, gt1)
        out1 = self.dropout1(out1)

        out_net2 = self.conv2A(out1)
        gt2 = self.gate2(out1)

        out2 = torch.mul(out_net2, gt2)
        out2 = self.dropout1(out2)

        if self.activation:
            out = self.activation(out2 + res)
        else:
            out = out2 + res
        # res = x if self.downsample is None else self.downsample(x)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """

        :param num_inputs: number of inputs (temporal part)
        :param num_channels: list of number of channels (spatial/current) for each convolutional layer
        :param kernel_size: default 2, keep it this way as it is good for power of 2 layers
        :param dropout:
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NLPGatedConv1DColumnV1(nn.Module):
    """
    Represents one Convolutional column from an input, starting with a kernel size and then
    all the next layers are a kernel 3x1
    At the end there is an attention layer

    The network is composed of 4 big layers:
    input layer with one convolution size and stride
    3 big layers, each with 2 sublayers, at the end of each big layer the dimension is diminished but there are more
    channels to the output


    Assume input dimension is an embedding of 1024 x c_in with embedding dimension c_in

    """
    def __init__(self, c_in, c_out=[64, 64, 128, 128],
                 kernel_size=3, hid_kernel_size=3,
                 # res_layers=3,
                 # sub_res_layers=3,
                 stride=1, hid_stride=1,
                 dropout=0.2,
                 activation="relu"):
        """

        :param c_in:
        :param c_out1: number of output channels of each big convolutional layer
        :param kernel_size: the first kernel size, all the others are size 2
        :param res_layers:
        :param stride: defaults to 1 but might be good to do something like min(1, kernel_size // 3)
        :param dropout:
        :param activation:
        """
        super(NLPGatedConv1DColumnV1, self).__init__()

        # assert len(c_out) == res_layers

        self.leftpad = nn.ConstantPad1d(kernel_size - 1)
        # first conv layer, the one with big kernel
        self.conv1 = weight_norm(nn.Conv1d(c_in, c_out[0], kernel_size, stride=stride))

        # now come the Residual Gated Convolutional Blocks

        #############################
        # First block of convolutions
        b1_gsl1 = GatedConv1DBlock(c_out[0], c_out[1], hid_kernel_size, dropout, activation)
        b1_gsl2 = GatedConv1DBlock(c_out[1], c_out[1], hid_kernel_size, dropout, activation)
        # cut input dimension by 2
        b1_maxp = nn.MaxPool1d(3, stride=2)
        #############################
        # Second block of convolutions
        b2_gsl1 = GatedConv1DBlock(c_out[1], c_out[2], hid_kernel_size, dropout, activation)
        b2_gsl2 = GatedConv1DBlock(c_out[2], c_out[2], hid_kernel_size, dropout, activation)
        # cut input dimension by 2
        b2_maxp = nn.MaxPool1d(3, stride=2)
        # dimension is cut by 4 already
        #############################
        # Third block of convolutions
        b3_gsl1 = GatedConv1DBlock(c_out[2], c_out[3], hid_kernel_size, dropout, activation)
        b3_gsl2 = GatedConv1DBlock(c_out[3], c_out[3], hid_kernel_size, dropout, activation)
        # cut input dimension by 2
        b3_maxp = nn.MaxPool1d(3, stride=2)
        # dimension got cut by 8 now
        self.conv_col = nn.Sequential(b1_gsl1, b1_gsl2, b1_maxp,
                                      b2_gsl1, b2_gsl2, b2_maxp,
                                      b3_gsl1, b3_gsl2, b3_maxp
                                      )
        #############################
        # Now is Attention Layer

        # And now the output activation

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)