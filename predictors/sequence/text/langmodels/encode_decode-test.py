import torch
import torch.nn as nn
# good that PyTorch v1.3.0+ has Transformers already implemented
from torch.nn.modules.transformer import Transformer
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
import faiss


class UTF8Code(nn.Module):
    def __init__(self, utf8codebook, idx2char, char2idx):
        """

        :param utf8codebook:
        :param idx2char:
        :param char2idx:
        """
        # self._codebook = utf8codebook
        self._index = faiss.IndexFlatL2(utf8codebook)
        # faiss.IndexBin
        # self._index = faiss.GpuIndexFlatL2(utf8codebook)
        # faiss.IVFL
        self._idx2char = idx2char
        self._char2idx = char2idx

    def idx2char(self, idxs):
        pass

    def char2idx(self, chars):
        pass

    def embed2idx(self, embeds):
        pass

    def forward(self, x):
        # this function basically calls embed2idx
        # x should be (batch size, sequence width, embedding)
        return self.embed2idx(x)


class UTF8Embedding(nn.Module):
    def __init__(self, utf8codebook, transpose=False, lin_layers=None):
        """
        Embedding Layer for UTF-8 based on pre-computed weights

        If the linear layers are created the output size of the embedding is the one of the last linear layer.
        By default there are no linear layers and the output of the embedding is the one given as input

        :param utf8codebook: Codebook containing the entire index to code matrix
        :param transpose: if the output should be transposed such as if True the result is shape
                          (batch size, embedding, sequence width) if False (default) shape is
                          (batch size, sequence width, embedding)
        :param lin_layers: iterable with size of the output of each linear layer, examples: [64], [256,64]
        """
        self._transpose = transpose
        super(UTF8Embedding, self).__init__()
        codebook_shape = utf8codebook.shape
        self.embeds = nn.Embedding(*codebook_shape)
        self.embeds.weight.data.copy_(torch.from_numpy(utf8codebook))
        self.embeds.weight.requires_grad_(False)
        self.lin = nn.ModuleList()
        self.embedding = nn.ModuleList()
        if lin_layers is not None:
            prev_dim = codebook_shape[1]
            for dim in lin_layers:
                lin = nn.Linear(prev_dim, dim)
                prev_dim = dim
                self.lin.append(lin)
        self.embedding.append(self.embeds)
        self.embedding.extend(self.lin)

    @property
    def transpose(self):
        return self._transpose

    @property
    def transpose(self, transpose):
        self._transpose = transpose

    def forward(self, x):
        # (batch size, sequence-width)
        ret = self.embedding(x)  # (batch size, sequence width[values]) -> # (batch size, sequence width, embedding)
        if self._transpose:
            ret = ret.transpose(1, 2)  # (batch size, embedding, sequence width)
        return ret


class UTF8AttentionalEmbedding(UTF8Embedding):
    def __init__(self, utf8codebook, layers=2, nheads=4, ffdim=1024, outdim=32):
        """
        Embedding Layer for UTF-8 based on pre-computed weights, it adds attentional and linear layers
        to be able to modify the embedding dimension.
        This class can be used later frozen to recompute the embeddings and use UTF8Embedding directly
        :param utf8codebook: Codebook containing the entire index to code matrix
        :param layers: number of attention layers (channel-wise, for each character)
        :param nheads: number of heads of the attentional layers
        :param ffdim: feed forward dimension of the attentional linear layer
        :param outdim: output dimension of the code
        """

        # dimension would be:
        # 1024,
        super(UTF8AttentionalEmbedding, self).__init__(utf8codebook)
        # compute the input weights
        codebook_shape = self.embeds.weight.shape
        # shape is [len codebook, dim embedding=64]
        # assert is here so if code changes I know (this should be changed in the future)
        assert codebook_shape[1] == 64
        self.att = nn.ModuleList()
        for i in range(layers):
            att = TransformerEncoderLayer(codebook_shape[1], nheads, dim_feedforward=ffdim)
            self.att.append(att)

        self.lin = nn.Linear(codebook_shape[1], outdim)
        self.fw = nn.ModuleList()
        # pre-coded / computed embeddings to (normally dim 64)
        self.fw.append(self.embeds)
        self.fw.extend(self.att)
        self.fw.append(self.lin)

    def forward(self, x):
        # (batch size, values)
        ret = self.fw(x)  # (batch size, sequence width[values]) -> # (batch size, sequence width, embedding)
        if self._transpose:
            ret = ret.transpose(1, 2)  # (batch size, embedding, sequence width)
        return ret


class UTF8Decoder(nn.Module):
    def __init__(self, utf8codebook):
        # dimension would be:
        # 1024,
        super(UTF8Decoder, self).__init__()
        self.embeds = nn.Embedding(*utf8codebook.shape)
        self.embeds.weight.data.copy_(torch.from_numpy(utf8codebook))
        self.embeds.weight.requires_grad(False)

    pass


class EncodeDecodeTest(nn.Module):
    pass


def main():
    pass
