from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

from fairseq.modules.dynamic_convolution import DynamicConv1dTBC
from torch.nn.modules.transformer import TransformerEncoderLayer
from fb_dynamicconv.dynamic_convolution import DynamicConv
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
            cnv = weight_norm(nn.Conv1d(t_c_in, c_out, kernel_size, padding=(kernel_size - 1) // 2, groups=groups))
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
                 activation="relu", gating_activation="sigmoid", use_residual=False):
        """
        :param c_in: # input channels
        :param c_out: # output channels
        :param kernel_size:
        :param nlayers: number of layers to use in this stack
        :param stride:
        :param dropout:
        :param activation: if None then no activation is done
        :param gating_activation: activation layer type for the gating mechanism
        :param use_residual: If residual should be added to the module. Default False (each sub-block uses already a residual
        """
        super(GatedConv1DBlock, self).__init__()
        self.use_residual = use_residual
        self.convresid = None

        if c_in != c_out and self.use_residual:
            self.use_proj = 1
            self.convresid = weight_norm(nn.Conv1d(c_in, c_out, 1))  # downsample for residual connection if needed
        else:  # if c_in == c_out or not use_residual:
            self.use_proj = 0

        self.convs = nn.ModuleList()
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
        ret = x
        # print("GatedBlock: ", x.shape)
        ret = self.network(x)
        # for cnv in self.convs:
        #     print(res.shape)
        #     ret = cnv(ret)
        if self.use_proj and self.use_residual:  # if in_c != out_c, need to change size of residual
            res = self.convresid(res)
        if self.use_residual:
            ret = ret + res
        return ret


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
        # gating unit
        self.conv1B = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=stride))
        self.gatingActiv1B = nn.Sigmoid()  # get_activation_fn(gating_activation)

        self.gate1 = nn.Sequential(self.conv1B, self.gatingActiv1B)

        self.dropout1 = nn.Dropout(dropout)
        self.conv2A = weight_norm(nn.Conv1d(c_out, c_out, kernel_size, stride=stride))
        # gating unit
        self.conv2B = weight_norm(nn.Conv1d(c_out, c_out, kernel_size, stride=stride))
        self.gatingActiv2B = nn.Sigmoid()  # get_activation_fn(gating_activation)

        self.dropout2 = nn.Dropout(dropout)

        self.gate2 = nn.Sequential(self.conv2B, self.gatingActiv2B)

        self.activation = get_activation_fn(activation)
        self._norm = nn.LayerNorm(c_out)  # nn.BatchNorm1d(c_out)
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
        # print(1, res.shape)
        # residual connection channel dimension adaptation
        x = self.leftpad(x)
        # print(2, x.shape)
        out_net1 = self.conv1A(x)
        gt1 = self.gate1(x)
        # print(3, x.shape, res.shape, out_net1.shape, gt1.shape)

        out1 = torch.mul(out_net1, gt1)
        out1 = self.dropout1(out1)
        # print(4, out1.shape)

        out1 = self.leftpad(out1)
        out_net2 = self.conv2A(out1)
        # print(4, out1.shape, out_net2.shape)
        gt2 = self.gate2(out1)
        # print(5, out_net2.shape, gt2.shape)

        out2 = torch.mul(out_net2, gt2)
        out2 = self.dropout1(out2)
        # print(6, out2.shape)

        if self.use_proj:  # if in_c != out_c, need to change size of residual
            # print("convresid")
            res = self.convresid(res)
        # print(1, res.shape)

        if self.activation:
            out = self.activation(out2 + res)
        else:
            out = out2 + res
        out = self._norm(out)
        # res = x if self.downsample is None else self.downsample(x)
        # print(7, out.shape)
        return out


class ConvLinBlock(nn.Module):
    """
    Convolutional + Linear Layer, takes a pre-trained convolutional block as input and ouputs 2 vectors:
    The output from the original convolutional block PLUS the output of the linear layer
    """

    def __init__(self, conv_block, in_conv_channels, lin_channels=64,
                 in_conv_dim=1024, conv_proj_dim=128, in_lin_dim=128, linear_dims=[1024, 128, 128],
                 dropout=0.3, activation="gelu", residual=True):
        """
        """
        super(ConvLinBlock, self).__init__()

        self.conv_block = conv_block  # pre-trained convolutional block
        self.conv_adapt = nn.Conv1d(in_conv_channels, lin_channels, 1)  # adapt number of channels for linear layers
        self.lin_adapt = nn.Linear(in_conv_dim, conv_proj_dim)  # adapt (project) number of samples for linear layers
        lin = []
        for ind, od in zip(([conv_proj_dim + in_lin_dim] + linear_dims)[:-1], linear_dims):
            lin.append(nn.Linear(ind, od))
        self.lin = nn.Sequential(*lin)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lin_channels, linear_dims[-1])  # normalize over the last 2 dims: [channels, seq]
        self.activation = get_activation_fn(activation)
        self.residual = residual

    def forward(self, x_conv, x_lin):
        # x_conv = [Batches, channels, seq_len]
        # x_lin = [Batches, channels, seq_len]
        # apply convolutional block
        x_conv = self.conv_block(x_conv)
        # adapt number of channels for linear
        x_conv = self.conv_adapt(x_conv)
        # adapt(project) sequence length to linear input layer
        x = self.lin_adapt(x_conv)
        # concatenate projection from convolutional layer with previous (linear) column output
        # over sequence length dimension (the last now)
        x = torch.cat([x, x_lin], dim=-1).contiguous()
        # apply linear over projection
        x = self.linear(x)
        x = self.activation(x)
        # residual
        if self.residual:
            # TODO find out if only the residual from the linear column or should add the convolutional part too
            x = x + x_lin
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x_conv, x


class ConvAttBlock(nn.Module):
    """
    Convolutional + Linear Layer, takes a pre-trained convolutional block as input and ouputs 2 vectors:
    The output from the original convolutional block PLUS the output of the linear layer
    """

    def __init__(self, conv_block, in_conv_channels, lin_channels=96,
                 in_conv_dim=1024, conv_proj_dim=256, att_layers=2, att_dim=256,
                 att_encoder_heads=8, att_encoder_ff_embed_dim=1024,
                 dropout=0.1, att_dropout=0.1, activation=None, residual=True):
        """
        """
        super(ConvAttBlock, self).__init__()

        self.conv_block = conv_block  # pre-trained convolutional block
        self.conv_adapt = nn.Conv1d(in_conv_channels, lin_channels, 1)  # adapt number of channels for linear layers
        self.lin_adapt = weight_norm(nn.Linear(in_conv_dim, conv_proj_dim))  # adapt (project) number of samples for next
        self.att_adapt = weight_norm(nn.Linear(conv_proj_dim + att_dim, att_dim))  # adapt (project) number of samples for att layers
        _att = []
        for i in range(att_layers):
            att = TransformerEncoderLayer(att_dim, att_encoder_heads, att_encoder_ff_embed_dim, att_dropout, "gelu")
            _att.append(att)
        # print("att layers = ", len(self._att))
        self.att = nn.Sequential(*_att)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.residual = residual

    def forward(self, x_conv, x_att):
        # x_conv = [Batches, channels, seq_len]
        # x_lin = [Batches, channels, seq_len]
        # apply convolutional block
        x_conv = self.conv_block(x_conv)
        # adapt number of channels for linear
        # print("1 model blocks ", x_conv.shape, x_att.shape)
        x = self.conv_adapt(x_conv)
        # print("2 model blocks ", x_conv.shape, x_att.shape, x.shape)
        # adapt(project) sequence length to linear input layer
        x = self.lin_adapt(x)
        # concatenate projection from convolutional layer with previous (linear) column output
        # over sequence length dimension (the last now)
        # print("3 model blocks ", x_conv.shape, x_att.shape, x.shape)
        x = torch.cat([x, x_att], dim=-1).contiguous()
        # project concatenation for Attention layer (attention layers are same input and output dimension)
        # print("4 model blocks ", x_conv.shape, x_att.shape, x.shape)
        x = self.att_adapt(x)
        # print("5 model blocks ", x_conv.shape, x_att.shape, x.shape)
        # apply attention over projection
        x = self.att(x)
        # print("6 model blocks ", x_conv.shape, x_att.shape, x.shape)
        if self.activation:
            x = self.activation(x)
        # residual
        if self.residual:
            # TODO find out if only the residual from the linear column or should add the convolutional part too
            x = x + x_att
        x = self.dropout(x)
        # x_conv = [Batches, channels, seq_len]
        # x_lin = [Batches, channels, seq_len]
        return x_conv, x


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

        pass


