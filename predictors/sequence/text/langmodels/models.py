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
# from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer
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


###


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
        self.fib_positions = torch.from_numpy(get_coord_emb(shape=(time_dim, fib_coord_channels), fibinit=6))
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
        self.encoder = Conv1DPartOfSpeechEncoder()  # use all default values
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
        conv_col = Conv1DPartOfSpeechEncoder()  # use all default values
        self.encoder = GatedConv1DPartOfSpeechEncoder(conv_col, return_all_layers=False)
        self.decoder = LinearUposDeprelDecoder()

        self.network = GenericNet(self.embeds, self.encoder, self.decoder)

    def forward(self, x):
        ret = self.embeds(x).float()
        ret = self.network(x)
        return ret


class Conv1DPartOfSpeechEncoder(nn.Module):
    def __init__(self, nchannels_in=[64, 128, 256, 512, 256],
                 nchannels_out=[128, 256, 512, 256, 96],
                 kernels=[3, 3, 3, 3, 3],
                 nlayers=[6, 6, 4, 4, 3],
                 groups=[1, 4, 8, 4, 1],  # allow more specialization of the block with most features
                 dropout=0.1,
                 activation="relu"
                 ):
        super(Conv1DPartOfSpeechEncoder, self).__init__()
        assert len(nchannels_in) == len(nchannels_out) == len(nlayers) == len(kernels)
        # store each block in a list so I can return each layer separately for other kind of processing

        self.convs = nn.ModuleList()
        for inc, outc, k, l, g in zip(nchannels_in, nchannels_out, kernels, nlayers, groups):
            cnv = Conv1DBlock(c_in=inc, c_out=outc, kernel_size=k, nlayers=l,
                              dropout=dropout, groups=g, activation=activation)
            self.convs.append(cnv)

    def forward(self, x):
        rets = []
        ret = x
        for cnv in self.convs:
            ret = cnv(ret)
            rets.append(ret)

        return rets


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

    def __init__(self, lin_in_dim=96, lin_hidd_dim=768,
                 upos_dim=18, deprel_dim=278,  # the number of features of UPOS and DEPREL in Conllu files
                 ):
        super(LinearUposDeprelDecoder, self).__init__()
        self.upos_dim = upos_dim
        self.deprel_dim = deprel_dim
        lin_out = upos_dim + deprel_dim  # commented while I make it work, TODO add it back
        # lin_out = upos_dim

        self.linears = nn.Sequential(
            weight_norm(nn.Linear(lin_in_dim, lin_hidd_dim)),
            weight_norm(nn.Linear(lin_hidd_dim, lin_out)),
            # nn.LayerNorm()
        )

    def forward(self, x):
        # (batch size, embedding, sequence length)
        ret = x
        # transpose to work channel-wise for the last decoding part
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
