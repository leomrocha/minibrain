
# from .fb_models import TransformerSeqLayer
from fairseq.t.transformer_layer import MultiheadAttention, TransformerDecoderLayer, TransformerEncoderLayer
from .utils.tools import *
from .decanlp_common import *

from .model_blocks import *


class FFColNet(nn.Module):
    def __init__(self, utf8codes, input_size=1024, batch_size=128,
                 in_channel_size=324,
                 embed_size=30, hid_size=256, output_size=128, layers=3
                 ):
        # dimension would be:
        # 1024,
        super(FFColNet, self).__init__()
        self.embeds = nn.Embedding(*(utf8codes.shape))
        self.embeds.weight.data.copy_(torch.from_numpy(utf8codes))
        # self.dense_embed = nn.Linear(input_size,)
        # self.relu1 = nn.ReLU()
        #
        # self.lin1 = nn.Linear(in_channel_size, hid_size)
        self.lin1 = nn.Linear(input_size, hid_size)
        self.tanh1 = nn.Tanh()
        self.lin2 = nn.Linear(hid_size, hid_size)
        self.tanh2 = nn.Tanh()
        self.lin3 = nn.Linear(hid_size, output_size)
        # self.lin3 = nn.Linear(hid_size, in_channel_size)

        # low and high bytes encodings for the multihot encoder
        self.actv_out = nn.Sigmoid()  # gelu
        # self.outb1 = nn.Softmax()
        # self.outb2 = nn.Softmax()
        self.net = nn.Sequential(self.lin1, self.tanh1,
                                 self.lin2, self.tanh2,
                                 self.lin3)

    def forward(self, x):
        x = self.embeds(x).transpose(1, 2)
        ret = self.net(x)
        # redimension so sigmoid is applied per channel and we can decode each output separately
        # ret = ret.transpose()
        ret = self.actv_out(ret).transpose(1, 2)
        return ret


class MixedConvLinearColumns(nn.Module):
    """

    """
    def __init__(self, utf8codes,
                 channels=[324, 64, 128, 128, 324],  # channels for blocks
                 b_layers=[5, 5, 5],  # number of layers for each bloc
                 c_linears=[324, 8, 8, 8, 324],  # in and out channels of the linear layers inputs
                 lin_in_dim=[1024+128, 512+128, 256+128, 128+128],  # max span allowed as input for each layer (must be <= than the available input)
                 lin_out_dim=[128, 128, 128, 128],
                 first_lin_size=1024,
                 first_k_sizes=[3, 5, 7, 9, 11, 15, 25], kernel_size=3,
                 dropout=0.3, groups=4):
        super(MixedConvLinearColumns, self).__init__()
        c_in = channels[:-1]
        c_out = channels[1:]
        c_lin_in = c_linears[:-1]
        c_lin_out = c_linears[1:]
        assert c_in[1:] == c_out[:-1]
        # assert len(c_out) == res_layers
        self.embeds = nn.Embedding(*(utf8codes.shape))
        self.embeds.weight.data.copy_(torch.from_numpy(utf8codes))

        # input convolution layer is the one who adapts the input for the following columns
        self.conv0 = nn.ModuleList()
        for k in first_k_sizes:
            cnv = nn.Conv1d(c_in[0], c_out[0], k, padding=(k - 1) // 2)
            # cnv = nn.Conv1d(c_in[0], c_out[0], k, padding=(k - 1) // 2, stride=(k - 1) // 2)
            self.conv0.append(cnv)
        self.conv0adapt = nn.Conv1d(len(first_k_sizes) * c_out[0], c_out[0], 1)
        self.drop0 = nn.Dropout(dropout)

        self.lin0 = nn.Linear(first_lin_size, lin_out_dim[0])

        self.lin_layers = nn.ModuleList()
        self.convs1x1_lin = nn.ModuleList()
        self.convs1x1_cnv = nn.ModuleList()

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        self.maxpool_blocks = nn.ModuleList()

        for cin, cout, lays in zip(c_in[1:], c_out[1:], b_layers):
            cnv = Conv1DBlock(cin, cout, kernel_size, lays, dropout, groups)
            mp = nn.MaxPool1d(2, stride=2)
            self.maxpool_blocks.append(mp)
            self.conv_blocks.append(cnv)
        # self.convolutions = nn.Sequential(*self.conv_blocks)
        # linear layers
        for cin, clin_in, clin_out, in_lin, out_lin in zip(c_out, c_lin_in, c_lin_out, lin_in_dim, lin_out_dim):
            # print("creating mixing layers ",cin, clin_in, clin_out, in_lin, out_lin)
            lin = MixedConvLinearColumns._get_lin_mix(cin, clin_in, clin_out, in_lin, out_lin, dropout)
            self.lin_layers.append(lin[0])
            self.convs1x1_cnv.append(lin[1])
            self.convs1x1_lin.append(lin[2])
        #
        self.last_lin = nn.Linear(lin_out_dim[-1], lin_out_dim[-1])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()

    @staticmethod
    def _get_lin_mix(c_in, c_lin_in, c_lin_out, lin_in_dim, lin_out_dim, dropout):
        # channel adapt from previous conv output to the linear cat
        conv1x1_1 = nn.Conv1d(c_in, c_lin_out, kernel_size=1)
        # channel adapt previous linear output to the linear cat
        conv1x1_2 = nn.Conv1d(c_lin_in, c_lin_out, kernel_size=1)
        # linear input from concatenation
        lin_1a = nn.Linear(lin_in_dim, lin_out_dim)
        # 'hidden' linear layer (could be changed by a convolutional one...but as it is small dimension ... )
        lin_1b = nn.Linear(lin_out_dim, lin_out_dim)
        drop_1 = nn.Dropout(dropout)
        tanh_1 = nn.Tanh()
        lin1 = nn.Sequential(lin_1a, lin_1b, drop_1, tanh_1)

        return lin1, conv1x1_1, conv1x1_2

    def forward(self, x):
        # print("1 mixcol", x.shape, x.dtype)
        emb = self.embeds(x).transpose(1, 2)
        print("2 mixcol", emb.shape)
        cnvs = [c(emb) for c in self.conv0]
        #  (batch_size, one_hot, sequence_width)
        cnv_col = torch.cat(cnvs, dim=1)
        print("2b mixcol", cnv_col.shape)
        cnv_col = self.conv0adapt(cnv_col)
        print("3 mixcol", cnv_col.shape)
        cnv_col = self.drop0(cnv_col)
        # print("4 mixcol", cnv_col.shape)
        cnv_blocks = [cnv_col]
        for conv, maxp in zip(self.conv_blocks, self.maxpool_blocks):
            cnv_col = conv(cnv_col)
            # print("5 mixcol", cnv_col.shape)
            cnv_col = maxp(cnv_col)
            # print("6 mixcol", cnv_col.shape)
            # save the result of the conv block for the attention layers
            cnv_blocks.append(cnv_col)
        cnv_col = self.dropout(cnv_col)
        # print("7 mixcol", cnv_col.shape)
        cnv_col = self.activation(cnv_col)
        # print("8 mixcol", cnv_col.shape)
        # cnv_col = cnv_col.transpose(1, 2)
        mix_col = self.lin0(emb)
        # print("9 mixcol", mix_col.shape)
        # Mixed Linear layers must be processed after the relative dependent convolutional parts
        # print(len(self.lin_layers), len(self.convs1x1_lin), len(self.convs1x1_cnv), len(cnv_blocks))
        for lin, cnv1, cnv2, out_cnv in zip(self.lin_layers, self.convs1x1_lin, self.convs1x1_cnv, cnv_blocks):
            # print("10a mixcol", out_cnv.shape)
            part1 = cnv2(out_cnv)
            # print("10 mixcol", part1.shape)
            part2 = cnv1(mix_col)
            # print("11 mixcol", part2.shape)
            cat1 = torch.cat((part1, part2), dim=2)
            # print("12 mixcol", cat1.shape)
            mix_col = lin(cat1)
            # print("13 mixcol", mix_col.shape)
        # last linear layer
        ret = self.last_lin(mix_col)
        # print("14 mixcol", ret.shape)
        # Sigmoid
        ret = self.activation(ret)
        # print("15 mixcol", ret.shape)
        ret = ret.transpose(1, 2)  # revert to the original input
        # print("16 mixcol", ret.shape)
        return ret


class MixedColsTest(nn.Module):
    def __init__(self):  # def __init__(self,  utf8codes):
        super(MixedColsTest, self).__init__()
        self.embeddings = nn.Embedding(1984, 32)
        # self.embeddings = nn.Embedding(*(utf8codes.shape))
        # self.embeddings.weight.data.copy_(torch.from_numpy(utf8codes))
        self.encoder = MixedConvAttentiveColumns()
        self.decoder = TransformerDecoder(128, 1, 256, 5, 0.3, True)
        # self.decoder = nn.Linear(128, 128)

    def forward(self, x):
        print("Embedding: ", x.shape)
        ret = self.embeddings(x).transpose(1, 2)
        print("Encoding: ", ret.shape)
        ret = self.encoder(ret)
        print("Decoding: ", ret.shape)
        ret = self.decoder(ret, ret).transpose(1, 2)
        # ret = self.decoder(ret).transpose(1, 2)
        print("Returning: ", ret.shape)
        return ret


class MixedConvAttentiveColumns(nn.Module):
    """

    """
    def __init__(self,  # utf8codes,
                 channels=[32, 64, 128, 128, 32],  # channels for blocks
                 b_layers=[5, 5, 5],  # number of layers for each bloc
                 c_attentive=[32, 64, 64, 64, 32],  # in and out channels of the linear layers inputs
                 att_in_dim=[1024+128, 512+128, 256+128, 128+128],  # max span allowed as input for each layer (must be <= than the available input)
                 # att_in_dim=[1024+256, 512+256, 256+256, 128+256],
                 # att_out_dim=[256, 256, 256, 256],
                 att_out_dim=[128, 128, 128, 128],
                 first_lin_size=1024,
                 first_k_sizes=[3, 5, 7, 9, 11, 15], kernel_size=3,
                 # first_k_sizes=[3, 5, 7, 9, 11, 15, 25], kernel_size=3,
                 n_heads=8, dropout=0.3, groups=8):
        super(MixedConvAttentiveColumns, self).__init__()
        c_in = channels[:-1]
        c_out = channels[1:]
        c_att_in = c_attentive[:-1]
        c_att_out = c_attentive[1:]
        # assert c_in[1:] == c_out[:-1]
        # assert len(c_out) == res_layers
        # self.embeds = nn.Embedding(*(utf8codes.shape))
        # self.embeds.weight.data.copy_(torch.from_numpy(utf8codes))

        # input convolution layer is the one who adapts the input for the following columns
        self.conv0 = nn.ModuleList()
        for k in first_k_sizes:
            cnv = nn.Conv1d(c_in[0], c_out[0], k, padding=(k-1)//2)
            # cnv = nn.Conv1d(c_in[0], c_out[0], k, padding=(k - 1) // 2, stride=(k - 1) // 2)
            self.conv0.append(cnv)
        self.conv0adapt = nn.Conv1d(len(first_k_sizes)*c_out[0], c_out[0], 1)
        self.drop0 = nn.Dropout(dropout)

        self.lin0 = nn.Linear(first_lin_size, att_out_dim[0])
        self.att0 = TransformerEncoderLayer(att_out_dim[0], n_heads, att_out_dim[0]*2, dropout)

        self.att_layers = nn.ModuleList()
        self.convs1x1_lin = nn.ModuleList()
        self.convs1x1_cnv = nn.ModuleList()

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        self.maxpool_blocks = nn.ModuleList()

        for cin, cout, lays in zip(c_in[1:], c_out[1:], b_layers):
            cnv = Conv1DBlock(cin, cout, kernel_size, lays, dropout, groups)
            mp = nn.MaxPool1d(2, stride=2)
            self.maxpool_blocks.append(mp)
            self.conv_blocks.append(cnv)
        # self.convolutions = nn.Sequential(*self.conv_blocks)
        # linear layers
        for cin, clin_in, clin_out, in_lin, out_lin in zip(c_out, c_att_in, c_att_out, att_in_dim, att_out_dim):
            # print("creating mixing layers ",cin, clin_in, clin_out, in_lin, out_lin)
            lin = MixedConvAttentiveColumns._get_att_mix(cin, clin_in, clin_out, in_lin, out_lin, n_heads, dropout)
            self.att_layers.append(lin[0])
            self.convs1x1_cnv.append(lin[1])
            self.convs1x1_lin.append(lin[2])
        #
        self.last_lin = nn.Linear(att_out_dim[-1], att_out_dim[-1])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()

    @staticmethod
    def _get_att_mix(c_in, c_lin_in, c_lin_out, lin_in_dim, lin_out_dim, n_heads, dropout):
        # channel adapt from previous conv output to the linear cat
        conv1x1_1 = nn.Conv1d(c_in, c_lin_out, kernel_size=1)
        # channel adapt previous linear output to the linear cat
        conv1x1_2 = nn.Conv1d(c_lin_in, c_lin_out, kernel_size=1)
        # linear input from concatenation to adapt to attention layer size
        lin_1a = nn.Linear(lin_in_dim, lin_out_dim)
        # attention layers
        # lin_1b = nn.(lin_out_dim, lin_out_dim)
        att_1a = TransformerEncoderLayer(lin_out_dim, n_heads, lin_out_dim*2, dropout)
        att_1b = TransformerEncoderLayer(lin_out_dim, n_heads, lin_out_dim*2, dropout)
        drop_1 = nn.Dropout(dropout)
        # activ_1 = nn.Tanh()  # = gelu  #
        # lin1 = nn.Sequential(lin_1a, att_1a, att_1b, drop_1, activ_1)
        lin1 = nn.Sequential(lin_1a, att_1a, att_1b, drop_1)

        return lin1, conv1x1_1, conv1x1_2

    def forward(self, x):
        print("1 mixcol", x.shape, x.dtype)
        # emb = self.embeds(x).transpose(1, 2)
        emb = x
        # print("2 mixcol", emb.shape)
        # cnv_col = self.conv0(emb)
        cnvs = [c(emb) for c in self.conv0]
        #  (batch_size, one_hot, sequence_width)
        cnv_col = torch.cat(cnvs, dim=1)
        cnv_col = self.conv0adapt(cnv_col)
        # print("3 mixcol", cnv_col.shape)
        cnv_col = self.drop0(cnv_col)
        print("4 mixcol", cnv_col.shape)
        cnv_blocks = [cnv_col]
        for conv, maxp in zip(self.conv_blocks, self.maxpool_blocks):
            cnv_col = conv(cnv_col)
            # print("5 mixcol", cnv_col.shape)
            cnv_col = maxp(cnv_col)
            # print("6 mixcol", cnv_col.shape)
            # save the result of the conv block for the attention layers
            cnv_blocks.append(cnv_col)
        cnv_col = self.dropout(cnv_col)
        print("7 mixcol", cnv_col.shape)
        cnv_col = self.activation(cnv_col)
        # print("8 mixcol", cnv_col.shape)
        # cnv_col = cnv_col.transpose(1, 2)
        # frist adapt with linear to attention column
        mix_col = self.lin0(emb)
        # First attention layer
        mix_col = self.att0(mix_col)
        print("9 mixcol", mix_col.shape)
        # Mixed Linear layers must be processed after the relative dependent convolutional parts
        # print(len(self.lin_layers), len(self.convs1x1_lin), len(self.convs1x1_cnv), len(cnv_blocks))
        for lin, cnv1, cnv2, out_cnv in zip(self.att_layers, self.convs1x1_lin, self.convs1x1_cnv, cnv_blocks):
            # print("10a mixcol", out_cnv.shape)
            part1 = cnv2(out_cnv)
            # print("10 mixcol", part1.shape)
            part2 = cnv1(mix_col)
            # print("11 mixcol", part2.shape)
            cat1 = torch.cat((part1, part2), dim=2)
            # print("12 mixcol", cat1.shape)
            mix_col = lin(cat1)
            # print("13 mixcol", mix_col.shape)
        # last linear layer
        ret = self.last_lin(mix_col)
        print("14 mixcol", ret.shape)
        # Sigmoid
        ret = self.activation(ret)
        print("15 mixcol", ret.shape)
        ret = ret.transpose(1, 2)  # revert to the original input
        print("16 mixcol", ret.shape)
        return ret


class ConvColumn(nn.Module):
    """
    """
    def __init__(self, utf8codes,
                 c_in=[128, 256, 512, 128], c_out=[128, 256, 512, 128],  # channels for blocks
                 b_layers=[3, 5, 5],  # number of layers for each bloc
                 first_k_size=3, kernel_size=3,
                 dropout=0.3, groups=4,
                 activation="sigmoid",  # "gelu"
                 ):
        super(ConvColumn, self).__init__()

        assert c_in[1:] == c_out[:-1]
        # assert len(c_out) == res_layers
        # input convolution layer is the one who adapts the input for the columns that follow
        self.conv0 = nn.Conv1d(c_in[0], c_out[0], first_k_size, padding=(kernel_size-1)//2, stride=(first_k_size-1)//2)
        self.drop0 = nn.Dropout(dropout)

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        # self.maxpool_blocks = nn.ModuleList()
        for cin, cout, lays in zip(c_in[1:], c_out[1:], b_layers):
            cnv = Conv1DBlock(cin, cout, kernel_size, lays, dropout, groups)
            # mp = nn.MaxPool1d(2, stride=2)
            self.conv_blocks.append(cnv)
            # self.maxpool_blocks.append(mp)
        self.convolutions = nn.Sequential(*self.conv_blocks)
        #
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)  # nn.Sigmoid()  # Better for multihot than relu

    def forward(self, x):
        ret = self.conv0(x)
        ret = self.drop0(ret)
        ret = self.convolutions(ret)
        ret = self.dropout(ret)
        ret = self.activation(ret)
        return ret


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
    def __init__(self, utf8codes, c_in, c_out=[64, 64, 128, 128],
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
        self.embeds = nn.Embedding(*(utf8codes.shape))
        self.embeds.weight.data.copy_(torch.from_numpy(utf8codes))

        pad = (kernel_size - 1)//2
        self.leftpad = nn.ConstantPad1d(pad, 0.)
        # first conv layer, the one with big kernel
        self.conv1 = weight_norm(nn.Conv1d(c_in, c_out[0], kernel_size, stride=stride))

        # now come the Residual Gated Convolutional Blocks

        #############################
        # First block of convolutions
        pad2 = (hid_kernel_size - 1) // 2
        leftpad1 = nn.ConstantPad1d(pad2, 0.)
        b1_gsl1 = GatedConv1DLayer(c_out[0], c_out[1], hid_kernel_size, dropout, activation)
        b1_gsl2 = GatedConv1DLayer(c_out[1], c_out[1], hid_kernel_size, dropout, activation)
        # cut input dimension by 2
        b1_maxp = nn.MaxPool1d(3, stride=2)
        #############################
        # Second block of convolutions
        leftpad2 = nn.ConstantPad1d(pad2, 0.)
        b2_gsl1 = GatedConv1DLayer(c_out[1], c_out[2], hid_kernel_size, dropout, activation)
        b2_gsl2 = GatedConv1DLayer(c_out[2], c_out[2], hid_kernel_size, dropout, activation)
        # cut input dimension by 2
        b2_maxp = nn.MaxPool1d(3, stride=2)
        # dimension is cut by 4 already
        #############################
        # Third block of convolutions
        leftpad3 = nn.ConstantPad1d(pad2, 0.)
        b3_gsl1 = GatedConv1DLayer(c_out[2], c_out[3], hid_kernel_size, dropout, activation)
        b3_gsl2 = GatedConv1DLayer(c_out[3], c_out[3], hid_kernel_size, dropout, activation)
        # cut input dimension by 2
        b3_maxp = nn.MaxPool1d(3, stride=2)
        # dimension got cut by 8 now
        self.conv_col1 = nn.Sequential(b1_gsl1, b1_gsl2, leftpad1, b1_maxp)
        self.conv_col2 = nn.Sequential(b2_gsl1, b2_gsl2, leftpad2, b2_maxp)
        self.conv_col3 = nn.Sequential(b3_gsl1, b3_gsl2, leftpad3, b3_maxp)
        #############################
        # Now is Attention Column Layer
        # TODO
        # Now the last attention layer that joins all together
        # And now the output activation

    def forward(self, x):
        print(11, x.shape)
        x = self.embeds(x).transpose(1, 2)
        print(22, x.shape)
        x = self.leftpad(x)
        # print(22, x.shape)
        out = self.conv1(x)
        print(33, out.shape)
        out = self.conv_col1(out)
        print(44, out.shape)
        out = self.conv_col2(out)
        print(55, out.shape)
        out = self.conv_col3(out)
        print(66, out.shape)
        res = x if self.downsample is None else self.downsample(x)
        print(77, res.shape)
        return self.relu(out)  # self.relu(out + res)


##########