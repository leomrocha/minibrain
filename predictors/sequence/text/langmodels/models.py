import numpy as np
import torch
import torch.nn as nn
import os
from .utils.position_coding import *
from .utils.tools import *
from .model_blocks import *
from .tcn import TemporalBlock, TemporalConvNet
# good that PyTorch v1.3.0+ has Transformers already implemented
# from torch.nn.modules.transformer import Transformer
# from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerDecoderLayer
import torch.nn.functional as F

from deprecation import deprecated


###
# Taken from https://github.com/facebookresearch/XLM  (though there are not many ways of doing this)
def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class NetContainer(nn.Module):
    """
    Generic module for handling the entire network
    The data flow and returned elements from
    """

    def __init__(self, utf8codes, encodernet, decodernet):
        """

        :param utf8codes: utf-8 pretrained (multi-hot) input encoder matrix (numpy)
        :param encodernet: network that creates the latent space vector
        :param decodernet: network that treats the latent space vector to the comparable output
        """
        super(NetContainer, self).__init__()
        self.embeddings = nn.Embedding(*utf8codes.shape)
        self.embeddings.weight.data.copy_(torch.from_numpy(utf8codes))
        self.encodernet = encodernet
        self.decodernet = decodernet

    def forward(self, x):
        # input as [Batch, Seq_Len, Embedding channels]
        # print(1, x.shape)
        x = self.embeddings(x)  # .transpose(1,2)
        # print(2, x.shape)
        # x still as [Batch, Seq_Len, Embedding channels]
        x = self.encodernet(x)
        # print(3, x.shape)
        # output as [Batch, Seq_Len, Embedding channels]
        x = self.decodernet(x)
        # print(4, x.shape)
        # output as [Batch, Seq_Len, Embedding channels]
        return x

    def save_checkpoint(self, path, base_name, cid="001"):
        """
        Saves each part of the model (positional embedding, channelwise linear, encoder, decoder) to different files
        with the base_name in the path
        :param path: path where to save the models
        :param base_name: the base part of the name
        :param cid: id to add to the model (for checkpointing)
        :return: the absolute path (including file name) to the ckeckpoint file
        """
        # check directory exists or create it
        if not os.path.exists(path):
            os.makedirs(path)
        bname = os.path.join(path, base_name)
        name = bname + "_" + cid
        name = name + ".state_dict.pth"
        data = {
            "embedding": self.embeddings.state_dict(),
            "encoder": self.encodernet.state_dict(),
            "decoder": self.decodernet.state_dict()
        }
        torch.save(data, name)
        return name

    def load_checkpoint(self, fpath):
        """
        Loads the model
        :param fpath: the file with path where to load the models from
        """
        checkpoint = torch.load(fpath)
        nets = [self.embeddings, self.encodernet, self.decodernet]
        keys = ["embedding", "encoder", "decoder"]
        # keys = ["lin_encoder", "encoder", "decoder"]
        for k, net in zip(keys, nets):
            net.load_state_dict(checkpoint[k])


class AbstractColumnNet(nn.Module):
    """
    Represents one column of the adaptation, consists of an encoder adapter from the Embedding to the shape that
    will be used later, a column of some Block type and a decoder.
    The decoder is NOT suposed to give a one-hot or multi-hot output, that part is left
    for the final decoder used for the final task

    """

    def __init__(self):
        super(AbstractColumnNet, self).__init__()
        self._input_encoder = None
        self._encoder = None
        self._decoder = None

    # @property
    # def input_coder(self):
    #     return self._input_encoder
    #
    # @property
    # def input_coder(self, encoder):
    #     self._input_encoder = encoder
    #
    # @property
    # def encoder(self):
    #     return self._encoder
    #
    # @property
    # def encoder(self, encoder):
    #     self._encoder = encoder
    #
    # @property
    # def decoder(self):
    #     return self._decoder
    #
    # @property
    # def decoder(self, decoder):
    #     self._decoder = decoder

    def forward(self, *input, **kwargs):
        raise NotImplementedError("Must subclass and override")


class Conv1DColNet(AbstractColumnNet):
    """

    """

    def __init__(self, in_dim=324, hidd_embed_dim=768, out_embed_dim=96,  # input Embedding adaptor channelwise
                 nchannels_in=[96, 128, 256, 512, 256, 128],  # Convolutional blocks
                 nchannels_out=[128, 256, 512, 256, 128, 96],
                 kernels=[3, 3, 3, 3, 3, 3],  # kernel_sizes for conv blocks
                 nlayers=[6, 5, 4, 5, 3, 3],  # number of layers for each block
                 groups=[1, 4, 8, 4, 2, 1],  # allow more specialization of the block with most features
                 dropout=0.1,  # dropout of each block
                 activation="relu",  # activation of each block (end of each)
                 out_in_dim=96, out_hidd_embed_dim=768,  # decoder channel_wise dimensions
                 transpose_output=False,
                 ):    # default conf = 2173824 trainable parameters
        super(Conv1DColNet, self).__init__()
        self.transpose_output = transpose_output

        self._input_encoder = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hidd_embed_dim)),
            weight_norm(nn.Linear(hidd_embed_dim, out_embed_dim)),
        )

        self._encoder = Conv1DEncoder(nchannels_in, nchannels_out, kernels, nlayers, groups, dropout, activation)

        self._decoder = nn.Sequential(
            weight_norm(nn.Linear(out_in_dim, out_hidd_embed_dim)),
            weight_norm(nn.Linear(out_hidd_embed_dim, out_embed_dim)),
        )

    def forward(self, x):
        # input as [Batch, Seq_Len, Embedding channels]
        # print(1, " encodernet ", x.shape)
        x = self._input_encoder(x)
        # print(2, " encodernet ", x.shape)
        x = x.transpose(1, 2).contiguous()
        # print(3, " encodernet ", x.shape)
        # output of input encoder as [Batch, Seq_Len, Embedding channels]
        x = self._encoder(x)
        # if self.transpose_output:
        x = x.transpose(1, 2).contiguous()
        # print(4, " encodernet ", x.shape, self._decoder)
        x = self._decoder(x)
        if self.transpose_output:
            x = x.transpose(1, 2).contiguous()
        # print(5, " encodernet ", x.shape)
        return x


class ConvAttColNet(AbstractColumnNet):
    def __init__(self, convcolnet,
                 in_dim=324, hidd_embed_dim=768, out_embed_dim=96,  # input Embedding adaptor channel-wise
                 in_seq_len=1024,
                 in_conv_channels=[128, 256, 512, 256, 128, 96], lin_channels=96,
                 in_conv_dim=1024, conv_proj_dim=192,  # 256,  # 192 because nvidia tensor units are [96x96]
                 att_layers=2, att_dim=192,  # 256,
                 att_encoder_heads=8, att_encoder_ff_embed_dim=1024,
                 dropout=0.1, att_dropout=0.1, activation=None, residual=True,
                 out_in_dim=96, out_hidd_embed_dim=768,  # 1024 # decoder channel_wise dimensions
                 out_seq_len=384,  # 512,  # output sequence length, maximum attention that will appear there
                 # out_seq_len=512,  # output sequence length
                 transpose_output=False,
                 ):  # default conf = 12838560 trainable parameters
        super(ConvAttColNet, self).__init__()

        self.transpose_output = transpose_output
        self._convcolnet = convcolnet
        self._input_encoder = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hidd_embed_dim)),
            weight_norm(nn.Linear(hidd_embed_dim, out_embed_dim)),
        )
        # to merge both linear embedding inputs into a single network
        # FIXME the 2*out_embed_dim works for this parametrization but might (will) brake for other configurations
        # self._embed_chann_project = weight_norm(nn.Linear(2*out_embed_dim, out_embed_dim, False))  # no bias
        self._embed_chann_project = weight_norm(nn.Conv1d(2*out_embed_dim, out_embed_dim, 1))  # Conv1x1 projection
        self._embed_seq_project = weight_norm(nn.Linear(in_seq_len, conv_proj_dim, False))  # no bias

        self._blocks = nn.ModuleList()
        # get the blocks from the convcolnet to give as input to the ConvAttBlock
        # TODO fix this access because is UGLY and not good encapsulated way
        conv1dcol_blocks = self._convcolnet._encoder._blocks
        assert len(in_conv_channels) == len(conv1dcol_blocks)
        for convb, in_conv_chann in zip(conv1dcol_blocks, in_conv_channels):
            att = ConvAttBlock(convb, in_conv_chann, lin_channels, in_conv_dim, conv_proj_dim, att_layers, att_dim,
                               att_encoder_heads, att_encoder_ff_embed_dim, dropout, att_dropout, activation, residual)
            self._blocks.append(att)

        # TODO fix this ... somehow nicely, sequential does not deal with multiple inputs and outputs
        # self._encoder = nn.Sequential(self._convattblocks)
        # linear adaptor for final output receives the output of convolutional layer passed by maxpool1d
        self._out_lin_adapt = nn.Linear(in_conv_dim // 2, conv_proj_dim)
        # TODO TransformerDecoderLayer asks for memory parameter -> this is another thing ..
        # self._out_att = TransformerDecoderLayer(out_seq_len, att_encoder_heads, out_seq_len*4, att_dropout, "gelu")
        self._out_att = TransformerEncoderLayer(out_seq_len, att_encoder_heads, out_seq_len*4, att_dropout, "gelu")

        # channel-wise decoder
        self._decoder = nn.Sequential(
            weight_norm(nn.Linear(out_in_dim, out_hidd_embed_dim)),
            weight_norm(nn.Linear(out_hidd_embed_dim, out_embed_dim)),
        )

    def forward(self, x):
        # print("1 ConvAttColNet ", x.shape)
        # input as [Batch, Seq_Len, Embedding channels]
        # input for the convolutional block
        x_conv = self._convcolnet._input_encoder(x)
        x_att = self._input_encoder(x)
        # print("2 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        # take advantage of the pre-trained network and add a new set of elements
        x_att = torch.cat([x_conv, x_att], dim=-1)
        # project input for the rest of the network to work in the defined shape and transpose to work on time
        # print("3 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        x_att = x_att.transpose(1, 2).contiguous()
        x_conv = x_conv.transpose(1, 2).contiguous()
        # print("4 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        x_att = self._embed_chann_project(x_att)
        # print("5 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        # project sequence length for processing later
        x_att = self._embed_seq_project(x_att)
        # print("6 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        # x_conv and x_att -> [Batch, Embedding, Seq_Len]
        for block in self._blocks:
            # convolutions stay untouched by the new attention column, but gives input to each Attention block
            x_conv, x_att = block(x_conv, x_att)
        # print("7 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        x_conv = F.max_pool1d(x_conv, kernel_size=2)  # reduce dimension by 2 ... reduces computation
        # print("8 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        # concatenate over "time" to make the last decisions, attention might have more importance than the conv part?
        x_conv = self._out_lin_adapt(x_conv)
        # print("9 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape)
        ret = torch.cat([x_conv, x_att], dim=-1).contiguous()
        # now ... what should I do? a big linear network with a final attention one? here is where things get tricky
        # I'll then assume that I will only check the last N elements, where N is arbitrary by conf
        # print("10 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape, ret.shape)
        ret = self._out_att(ret)
        # print("11 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape, ret.shape)
        # now transpose
        # if self.transpose_output:
        ret = ret.transpose(1, 2).contiguous()  # need for the channel-wise decoder
        # print("12 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape, ret.shape)
        # output as [Batch, Seq_Len, Embedding channels]
        ret = self._decoder(ret)
        # print("13 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape, ret.shape)
        if self.transpose_output:
            ret = ret.transpose(1, 2).contiguous()
            # output as [Batch, Embedding channels, Seq_Len]
        # print("14 ConvAttColNet ", x_conv.shape, x_att.shape, x.shape, ret.shape)
        return ret


############################################################################
# OLD
############################################################################


class GenericNet(nn.Module):
    """
    Generic module for handling the entire network
    The data flow and returned elements from
    """

    def __init__(self, utf8encoder, encodernet, decodernet,
                 time_dim=1024,
                 in_enc_dim=64,
                 fib_coord_channels=22,
                 # hid_code_dim=1024,
                 # out_code_dim=128,
                 hid_code_dim=512,
                 out_code_dim=64,
                 max_sequence_len=1024
                 ):
        """

        :param utf8encoder: text encoder
        :param encodernet: network that creates the latent space vector
        :param decodernet: network that treats the latent space vector to a dense output
        :param output_decoder: decoder of the dense output to the format needed for supervised training or user output
            (should be mostly linear and [one|multi]-hot encoding)
        """
        super(GenericNet, self).__init__()
        self.utf8encoder = utf8encoder

        # TODO use these embeddings
        self.position_embeddings = nn.Embedding(max_sequence_len, in_enc_dim)
        create_sinusoidal_embeddings(max_sequence_len, in_enc_dim, out=self.position_embeddings.weight)
        # fib_positions are fixed by the sequence position and added as extra channels
        # self.fib_positions = torch.from_numpy(get_coord_emb(shape=(time_dim, fib_coord_channels), fibinit=6))
        # TODO take this out, and leave it for each encoder
        #  (as each of them might need some of this to see new things when incremental training)
        # adapt sinusoidal embeddings dimension to the desired size before the columns process the code
        self.lin_chann = nn.Sequential(
            # nn.Linear(in_enc_dim+fib_coord_channels, hid_code_dim),
            weight_norm(nn.Linear(in_enc_dim, hid_code_dim)),
            weight_norm(nn.Linear(hid_code_dim, out_code_dim)),
            # TODO check if activation and dropout is needed
        )
        self.encodernet = encodernet
        self.decodernet = decodernet

    def forward(self, x, positions=None):
        # (batch size, sequence width[values])
        txtcode = self.utf8encoder(x)
        # multihot, txtcode = self.utf8encoder(x)  # input encoder must return -> (multihot, dense)
        # (batch size, sequence length, embedding)  -> make sure the encoder does NOT transpose dims 1&2
        # add temporal coordinates
        # print(x.shape, txtcode.shape)
        bs, slen, semb = txtcode.shape
        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            # TODO check this assertion
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        txtcode = txtcode + self.position_embeddings(positions).expand_as(txtcode)
        # fibonacci absolute sequence positions
        # txtcode = torch.cat((txtcode, self.fib_positions), dim=-1).contiguous()
        # adapt number of embedding channels to the rest of the network
        txtcode = self.lin_chann(txtcode)
        # prepare for temporal processing
        txtcode = txtcode.transpose(1, 2).contiguous()  # permute and transpose are equivalent
        # (batch size, embedding, sequence length)
        latent = self.encodernet(txtcode)

        dec = self.decodernet(latent[-1])  # only last element of the encoded values, the others are for future use

        # return multihot, txtcode, positions, latent, dec
        return txtcode, positions, latent, dec

    def save_checkpoint(self, path, base_name, cid="001"):
        """
        Saves each part of the model (positional embedding, channelwise linear, encoder, decoder) to different files
        with the base_name in the path
        :param path: path where to save the models
        :param base_name: the base part of the name
        :param cid: id to add to the model (for checkpointing)
        :return: the absolute path (including file name) to the ckeckpoint file
        """
        # check directory exists or create it
        if not os.path.exists(path):
            os.makedirs(path)
        bname = os.path.join(path, base_name)
        name = bname + "_" + cid
        name = name + ".state_dict.pth"
        data = {
            "pos_embedding": self.position_embeddings.state_dict(),
            "lin_encoder": self.lin_chann.state_dict(),
            "encoder": self.encodernet.state_dict(),
            "decoder": self.decodernet.state_dict()
        }
        torch.save(data, name)
        return name

    def load_checkpoint(self, fpath):
        """
        Loads the model
        :param fpath: the file with path where to load the models from
        """
        checkpoint = torch.load(fpath)
        nets = [self.position_embeddings, self.lin_chann, self.encodernet, self.decodernet]
        keys = ["pos_embedding", "lin_encoder", "encoder", "decoder"]
        # keys = ["lin_encoder", "encoder", "decoder"]
        for k, net in zip(keys, nets):
            net.load_state_dict(checkpoint[k])
            # net.load_state_dict(checkpoint[k], strict=False)

    @deprecated
    def old_save_model(self, path, base_name, nid="001", save_statedict=True):
        """
        Saves each part of the model (positional embedding, channel-wise linear, encoder, decoder) to different files
        with the base_name in the path
        :param path: path where to save the models
        :param base_name: the base part of the name
        :param nid: id to add to the model (for checkpointing)
        :param save_statedict: if yes will save the statedict instead of the model (should be lighter)
        """
        # check directory exists or create it
        if not os.path.exists(path):
            os.makedirs(path)
        bname = os.path.join(path, base_name)
        bname = bname + "_" + nid
        pos_embed_name = bname + "_pos-embedding"
        lin_name = bname + "_lin_encoder"
        enc_name = bname + "_encoder"
        dec_name = bname + "_decoder"

        names = [pos_embed_name, lin_name, enc_name, dec_name]
        nets = [self.position_embeddings, self.lin_chann, self.encodernet, self.decodernet]
        for name, net in zip(names, nets):
            data = net.state_dict() if save_statedict else net
            name = name + ".state_dict" if save_statedict else name
            name = name + ".pth"
            torch.save(data, name)

    # @deprecated
    def old_load_model(self, path, base_name, saved_statedict=True):
        """
        Loads each part of the model (positional embedding, channelwise linear, encoder, decoder) from different files
        with the base_name in the path
        :param path: path where to load the models from
        :param base_name: the base part of the name
        :param saved_statedict: if yes will save the statedict instead of the model (should be lighter)
        """
        # check directory exists or create it
        if not os.path.exists(path):
            os.makedirs(path)
        bname = os.path.join(path, base_name)
        pos_embed_name = bname + "_pos-embedding"
        lin_name = bname + "_lin_encoder"
        enc_name = bname + "_encoder"
        dec_name = bname + "_decoder"

        names = [pos_embed_name, lin_name, enc_name, dec_name]
        # nets = [self.position_embeddings, self.lin_chann, self.encodernet, self.decodernet]
        nets = [self.position_embeddings, self.lin_chann, self.encodernet]
        for name, net in zip(names, nets):
            name = name + ".state_dict" if saved_statedict else name
            name = name + ".pth"
            net.load_state_dict(torch.load(name), strict=False)


@deprecated
class OLD_Conv1DPoS(nn.Module):
    """
    Part Of Speech, fixed network for testing purposes
    """

    def __init__(self, utf8codes):
        super(OLD_Conv1DPoS, self).__init__()
        # this time uses the already pre-computed UTF8 -> 64 dimensions encoder, make sure not to train it again
        with torch.no_grad():
            self.embeds = nn.Embedding(*(utf8codes.shape))
            self.embeds.weight.data.copy_(torch.from_numpy(utf8codes))

        # Encoder
        self.encoder = Conv1DEncoder()  # use all default values
        self.decoder = LinearUposDeprelDecoder()

        self.network = GenericNet(self.embeds, self.encoder, self.decoder)

    def forward(self, x):
        return self.network(x)


class GatedConv1DPoS(nn.Module):
    """
    Part Of Speech, fixed network for testing purposes
    """

    def __init__(self, utf8codes):
        super(GatedConv1DPoS, self).__init__()
        # this time uses the already pre-computed UTF8 -> 64 dimensions encoder, make sure not to train it again
        with torch.no_grad():
            self.embeds = nn.Embedding(*(utf8codes.shape))
            self.embeds.weight.data.copy_(torch.from_numpy(utf8codes))

        # Encoder
        conv_col = Conv1DEncoder()  # use all default values
        self.encoder = GatedConv1DPartOfSpeechEncoder(conv_col, return_all_layers=False)
        self.decoder = LinearUposDeprelDecoder()

        self.network = GenericNet(self.embeds, self.encoder, self.decoder)

    def forward(self, x):
        ret = self.embeds(x).float()
        ret = self.network(x)
        return ret


class Conv1DEncoder(nn.Module):
    def __init__(self, nchannels_in=[96, 128, 256, 512, 256],
                 nchannels_out=[128, 256, 512, 256, 96],
                 kernels=[3, 3, 3, 3, 3],
                 nlayers=[6, 5, 4, 5, 3],
                 groups=[1, 4, 8, 4, 1],  # allow more specialization of the block with most features
                 dropout=0.1,
                 activation="relu"
                 ):
        super(Conv1DEncoder, self).__init__()
        assert len(nchannels_in) == len(nchannels_out) == len(nlayers) == len(kernels)
        # store each block in a list so I can return each layer separately for other kind of processing
        self._blocks = nn.ModuleList()
        for inc, outc, k, l, g in zip(nchannels_in, nchannels_out, kernels, nlayers, groups):
            cnv = Conv1DBlock(c_in=inc, c_out=outc, kernel_size=k, nlayers=l,
                              dropout=dropout, groups=g, activation=activation)
            self._blocks.append(cnv)
        self.convs = nn.Sequential(*self._blocks)

    def forward(self, x):
        ret = self.convs(x)
        return ret


class GatedConv1DPartOfSpeechEncoder(nn.Module):
    def __init__(self, conv_column,  # the pretrained Conv1D Column from which will get intermediate results
                 conv1d_out=[128, 256, 512, 256, 96],  # output dimensions from the conv1d column
                 nchannels_in=[64, 128, 128, 128, 128, 192],  # Conv1D resampled + prev GatedConv channels
                 nchannels_out=[64, 64, 64, 64, 96, 96],
                 kernels=[3, 3, 3, 3, 3, 3],
                 nlayers=[3, 3, 3, 3, 3, 2],
                 dropout=0.1,
                 activation="gelu",
                 gating_activation="sigmoid",
                 return_all_layers=False  # if should return the result of all layers
                 ):
        super(GatedConv1DPartOfSpeechEncoder, self).__init__()
        self.return_all_layers = return_all_layers
        assert len(nchannels_in) == len(nchannels_out) == len(nlayers) == len(kernels)
        # store each block in a list so I can return each layer separately for other kind of processing

        self.conv1d_col = conv_column
        # downsampling from conv_column
        self.downsamples = nn.ModuleList()
        for din, dout in zip(conv1d_out, nchannels_out[:-1]):
            self.downsamples.append((nn.Conv1d(din, dout, kernel_size=1)))

        self.gconvs = nn.ModuleList()
        for inc, outc, k, l in zip(nchannels_in, nchannels_out, kernels, nlayers):
            cnv = GatedConv1DBlock(c_in=inc, c_out=outc, kernel_size=k, nlayers=l,
                                   dropout=dropout, activation=activation, gating_activation=gating_activation)
            self.gconvs.append(cnv)

    def forward(self, x):
        # get outputs from pre-trained conv1d col
        conv1d_ret = self.conv1d_col(x)
        # print(x.shape, len(conv1d_ret), len(self.downsamples))
        assert len(conv1d_ret) == len(self.downsamples) == len(self.gconvs) - 1
        # downsample for joining data with input from
        downsamples = []
        for ds, dta in zip(self.downsamples, conv1d_ret):
            downsamples.append((ds(dta)))
        # compute result of each for each gated block
        rets = []
        last_ret = self.gconvs[0](x)  # this is only when i == 0 i.e. when reading data from the input
        rets = [last_ret]
        for i, cnv in enumerate(self.gconvs[1:]):
            dta = torch.cat([downsamples[i], last_ret], dim=1)
            # print("merging shapes: ", i, downsamples[i].shape, last_ret.shape, dta.shape)
            last_ret = cnv(dta)
            if self.return_all_layers:
                rets.append(last_ret)
            else:  # saving memory (need it
                rets = [last_ret]  # this saves memory
        return rets


class TCNColumn(nn.Module):
    """
    Processes the input BxNSeqxchannels time and creates an output of Bx1xNseq
    """

    def __init__(self, in_chann=64, n_channels=[128, 128, 256, 64]):
        # what I want with the TCN is to be able to "summarize" the input to use it with the Transformer Layers
        # so, if the input is by default 1024x64, the output should be 128x64 (which is later concatenated with
        # the transformer column), this means, len [1024, 512, 256, 128] => 4 layers and 64 channels
        # I choose parameters to keep the number of parameters low as it will be used in each output from the Conv1D
        # blocks, the idea is to give the Transformer layers information about all the temporal output
        # in the current configuration one column is 412352 parameters
        # from the Conv1D blocks
        super(TCNColumn, self).__init__()
        self.net = TemporalConvNet(in_chann, n_channels)

    def forward(self, x):
        return self.net(x)


class LinearUposDeprelDecoder(nn.Module):
    """
    Decoding is Linear to make the encoding network do all the work so the embedding can be reused by other
    Tasks later
    """

    def __init__(self, lin_in_dim=96, lin_hidd_dim=960,
                 upos_dim=18, deprel_dim=278,  # the number of features of UPOS and DEPREL in Conllu files
                 transpose_input=False):
        super(LinearUposDeprelDecoder, self).__init__()
        self.upos_dim = upos_dim
        self.deprel_dim = deprel_dim
        self.transpose_input = transpose_input
        lin_out = upos_dim + deprel_dim  # commented while I make it work, TODO add it back
        # lin_out = upos_dim

        self.linears = nn.Sequential(
            weight_norm(nn.Linear(lin_in_dim, lin_hidd_dim)),
            weight_norm(nn.Linear(lin_hidd_dim, lin_out)),
            # nn.LayerNorm()
        )

    def forward(self, x):
        # (batch size, sequence length, embedding)
        ret = x
        # transpose to work channel-wise for the last decoding part
        if self.transpose_input:
            ret = ret.transpose(1, 2).contiguous()
        # (batch size, sequence length, embedding)
        ret = self.linears(ret)
        # apply Softmax per PoS characteristic
        # ret[:, :, :self.upos_dim] = F.softmax(ret[:, :, :self.upos_dim], dim=-1)  # upos decoding
        # upos = F.softmax(ret, dim=-1)  # upos decoding
        # upos = F.log_softmax(ret, dim=-1)  # upos decoding  # for NLL loss
        upos = F.log_softmax(ret[:, :, :self.upos_dim], dim=-1)  # upos decoding
        # ret[:, :, self.upos_dim:] = F.log_softmax(ret[:, :, self.upos_dim:], dim=-1)# deprel decoding (from upos position)
        deprel = F.log_softmax(ret[:, :, self.upos_dim:], dim=-1)  # deprel decoding (from upos position)
        # print("Lin decoding ", upos.shape, deprel.shape)
        return upos, deprel


class DynConvAttention(nn.Module):
    pass
