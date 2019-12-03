import numpy as np
import os
import pickle
import torch
import torch.nn as nn
# good that PyTorch v1.3.0+ has Transformers already implemented
from torch.nn.modules.transformer import TransformerEncoderLayer
import torch.nn.functional as F
import faiss

from .utils.tools import *


class UTF8CodeBook(nn.Module):
    def __init__(self, utf8codebook, idx2char, char2idx, k=1):
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
        self._k = k

    def idx2char(self, idxs):
        """
        :param idxs: a vector (numpy) containing the indices to convert to char
        :return: a vector of characters (numpy)
        """
        ret = np.vectorize(self._idx2char.get)(idxs)
        return ret

    def char2idx_np(self, chars):
        ret = np.vectorize(self._char2idx.get)(chars)
        return ret

    def char2idx_torch(self, chars):
        # TODO ???
        pass

    def embed2idx_np(self, embeds):
        _, indices = self._index.search(embeds, self._k)
        return indices

    def embed2idx_torch(self, embeds):
        _, indices = self._index.search(embeds, self._k)
        ret = torch.from_numpy(indices)
        return ret

    def forward(self, x):
        # this function basically calls embed2idx
        # x should be (batch size, sequence width, embedding)
        # TODO check this cause this may go in gpu
        return self.embed2idx_torch(x)


class UTF8Embedding(nn.Module):

    def __init__(self, utf8codebook, transpose=False, lin_layers=(512, 512, 64), activation=None, dropout=0.1):
        """
        Embedding Layer for UTF-8 based on pre-computed weights

        If the linear layers are created the output size of the embedding is the one of the last linear layer.
        By default there are no linear layers and the output of the embedding is the one given as input

        :param utf8codebook: Codebook containing the entire index to code matrix
        :param transpose: if the output should be transposed such as if True the result is shape
                          (batch size, embedding, sequence width) if False (default) shape is
                          (batch size, sequence width, embedding)
        :param lin_layers: iterable with size of the output of each linear layer, examples: [64], [256,64]
        :param activation: activation type
        :param dropout: for training dropout
        """
        self._transpose = transpose
        super(UTF8Embedding, self).__init__()
        codebook_shape = utf8codebook.shape
        self.embeds = nn.Embedding(*codebook_shape)
        self.embeds.weight.data.copy_(torch.from_numpy(utf8codebook))
        self.embeds.weight.requires_grad_(False)  # input embedding is fixed
        self.lin = nn.ModuleList()

        if lin_layers:
            prev_dim = codebook_shape[1]
            for dim in lin_layers:
                lin = nn.Linear(prev_dim, dim)
                # dropout = nn.Dropout(dropout)
                prev_dim = dim
                # self.lin.extend([lin, dropout])
                self.lin.append(lin)

        # self.embedding.append(self.embeds)
        # self.embedding.extend(self.lin)
        self.linear = nn.Sequential(*self.lin)
        self.activation = get_activation_fn(activation)

    @property
    def transpose(self):
        return self._transpose

    @property
    def transpose(self, transpose):
        self._transpose = transpose

    def forward(self, x):
        # (batch size, sequence-width)
        # print(x.shape, type(x), x.dtype)
        emb = self.embeds(x)
        dense = self.linear(emb)
        # ret = self.embedding(x)  # (batch size, sequence width[values]) -> # (batch size, sequence width, embedding)
        if self._transpose:
            dense = dense.transpose(1, 2)  # (batch size, embedding, sequence width)
        if self.activation:
            dense = self.activation(dense)
        # the pre-computed embedding is needed to compute loss and avoid having to decode.
        # As it is pre-computed and fixed (does NOT change) this avoids computation to get back to the index matrix
        # during the decoding step
        return emb, dense  # returns original pre-computed embedding plus the dense embedding


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
        fw = nn.ModuleList()
        # pre-coded / pre-computed embeddings to (normally dim 32)
        fw.append(self.embeds)
        fw.extend(self.att)
        fw.append(self.lin)
        self.fw = nn.Sequential(*fw)

    def forward(self, x):
        # (batch size, values)
        ret = self.fw(x)  # (batch size, sequence width[values]) -> # (batch size, sequence width, embedding)
        if self._transpose:
            ret = ret.transpose(1, 2)  # (batch size, embedding, sequence width)
        return ret


class UTF8Decoder(nn.Module):
    def __init__(self, utf8codebook_shape, lin_layers=(512, 512), activation="gelu",
                 seg_indices=(0, 4, 256+4, 64+256+4, 2*64 + 256+4, 3*64 + 256+4), dropout=0.1):
        """
        This modules decodes a given code, if the code_dim is given (an iterable not None) then a MLP decoder is created to adapt
        from the input dimension to the utf8codebook dimension
        :param utf8codebook_shape:  shape of the pre-computed codebook used
        :param lin_layers: if not None, creates a MLP with the given dimensions  for example[512, 512, 388] and after that
        a multiple Softmax is used for each segment on the code
        :param activation: intermediate activation function
        :param seg_indices: START and END indices of the segments on the multihot code, default for 4 segments code
        for 1 segment1 is: seg_indices=(0, 4, 256+4)
        for 2 segments is: seg_indices=(0, 4, 256+4, 64+256+4)
        for 3 segments is: seg_indices=(0, 4, 256+4, 64+256+4, 2*64 + 256+4)
        for 4 segments is: seg_indices=(0, 4, 256+4, 64+256+4, 2*64 + 256+4, 3*64 + 256+4)
        but with the default value works for every segment encoding as it has a verification, so don't touch
        """
        super(UTF8Decoder, self).__init__()
        # self.embeds = nn.Embedding(*utf8codebook.shape)
        # self.embeds.weight.data.copy_(torch.from_numpy(utf8codebook))
        # self.embeds.weight.requires_grad(False)
        # the decoder has to pass from a dimension that compresses all the information to several vectors that
        # each encode one part of the output that later will be mapped by the utf8codebook to an int
        # the mechanism is:
        # input -> adaptation to utf8codebook_shape by linear layers ->
        #       -> separation in segments ->
        #       -> Softmax ->
        #       -> concatenation
        # And later must be decoded to an integer index. This decoding to int
        # will be done in another class as it's a work on it's own and better separate it

        # Linear layers to adapt input latent to the codebook dimension
        self.lin = nn.ModuleList()
        if lin_layers:
            lin_layers = list(lin_layers) + [utf8codebook_shape[1]]  # add dimension
            for i in range(len(lin_layers)-1):
                lin = nn.Linear(lin_layers[i], lin_layers[i+1])
                # drop = nn.Dropout(dropout)
                # self.lin.extend([lin, drop])
                self.lin.append(lin)
        self.linear = nn.Sequential(*self.lin)
        self.activation = get_activation_fn(activation)
        # Linear layers to adapt dimension for the separated softmax
        #  precomputing dimensions and filtering
        sidx = np.array(seg_indices)
        sidx = sidx[sidx <= utf8codebook_shape[1]]  # filter out unused segments (dimension)
        ridx = np.roll(sidx, shift=-1)  # get the end index of each segment
        dims = ridx - sidx   # get segment dimensions
        dims = dims[0 < dims]  # this takes out the remaining problem dimensions (mainly 0)
        # creating the separated pre-softmax linear layers
        self.segments = nn.ModuleList()  # the order is important as it is the concatenation order
        for d in dims:
            lin = nn.Linear(utf8codebook_shape[1], d)
            self.segments.append(lin)

    def forward(self, x):
        # assumes input shape as:  (batch size, sequence width, embedding)
        # adapt input dimension to codebook
        ret = self.linear(x)
        if self.activation:
            ret = self.activation(ret)
        # process every segment part of the multihot-encoding
        segments = []
        for net in self.segments:
            seg = F.softmax(net(ret), dim=-1)
            # print("lin segment: ", net, seg.shape, seg.dtype)
            segments.append(seg)
        # concatenate results in axis to get to the embedding size:
        res = torch.cat(segments, dim=-1)
        # print("cat shape: ", res.shape, res.dtype)
        return res


class UTF8Autoencoder(nn.Module):
    def __init__(self, utf8codebook, dim=32):
        super(UTF8Autoencoder, self).__init__()
        self.encoder = UTF8Embedding(utf8codebook, lin_layers=(512, 512, dim))
        self.decoder = UTF8Decoder(utf8codebook.shape, lin_layers=[dim, 512, 512])

    def forward(self, x):
        emb, enc = self.encoder(x)
        dec = self.decoder(enc)
        return emb, dec

    def save_model(self, name, path):
        torch.save(self.encoder, os.path.join(path, "utf8Autoencoder_embedding_"+name+".pth"))
        torch.save(self.decoder, os.path.join(path, "utf8Autoencoder_decoder_"+name+".pth"))


def _load_pkl(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def _load_codebook(dir_path="./utf8-codes", segments=2):
    """
    Load the codebooks with the recommended naming from the utf8_encoder module
    This function is mainly for internal testing
    :param dir_path: directory where the pkl and numpy files are saved
    :return: a 5-tuple (code_matrix, txt2code, code2txt, txt2num, num2txt)
    """
    code_matrix = np.load(os.path.join(dir_path, "utf8_code_matrix_{}seg.npy".format(segments)))
    txt2code = _load_pkl(os.path.join(dir_path, "txt2code_{}seg.pkl".format(segments)))
    code2txt = _load_pkl(os.path.join(dir_path, "code2txt_{}seg.pkl".format(segments)))
    txt2num = _load_pkl(os.path.join(dir_path, "txt2num_{}seg.pkl".format(segments)))
    num2txt = _load_pkl(os.path.join(dir_path, "num2txt_{}seg.pkl".format(segments)))
    return code_matrix, txt2code, code2txt, txt2num, num2txt


def _prepare_overfit_batch(num2txt, batch_size):
    """
    The idea is to prepare the list of all the numbers in batches, the batches are randomly mixed to avoid issues.
    each batch contains:
    (batch size, seq width, index)  ??
    (batch size, index)  ??
    :param num2txt: numeric index 2 string conversion dictionary containing the entire vocabulary
    :return:
    """
    # assert type(num2txt) == 'dict'
    all_data = np.array(list(num2txt.keys()))
    all_data = all_data.reshape((-1, 1))
    # assume that we can hold all in memory
    arr = []
    for i in range(batch_size):
        data = np.copy(all_data)
        np.random.shuffle(data)
        arr.append(data.transpose())

    ret = np.stack(arr, axis=1)
    ret = ret.reshape(batch_size, -1)
    return ret


def train_overfit(model, optimizer, loss_function, batches, epoch, device, log_interval=10):
    train_loss = 0
    batch_loss = []
    batch_idx = 0
    for b in batches:
        tensor_data = torch.from_numpy(b).to(device).long()  #.double()  #.float()
        optimizer.zero_grad()
        # emb is obtained from the the pre-computed utf8codebook
        emb, res = model(tensor_data)
#         print(emb.shape,emb.dtype, res.shape, res.dtype)
        loss = loss_function(emb, res)
        loss.backward()
        train_loss += loss.data.item()  # [0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(batches),
                100. * batch_idx / len(tensor_data),
                train_loss / len(tensor_data)))
            batch_loss.append(train_loss)
        batch_idx += 1
    print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(batches)))
    return batch_loss


def test(model, test_data, loss_function, epoch, device):
    model.eval()
    test_loss = 0
    for d in test_data:
        tensor_data = torch.from_numpy(d).to(device)
        res = model(d)
        test_loss += loss_function(tensor_data, res).data.item()  # [0]

    test_loss /= len(test_data)
    print('epoch: {}====> Test set loss: {:.4f}'.format(epoch, test_loss))

# from https://discuss.pytorch.org/t/shuffling-a-tensor/25422/4
# TODO FIXME
# def _prepare_overfit_batch_torch(num2txt, batch_size):
#
#     t = torch.tensor(list(num2txt.keys()))
#     r = torch.randperm(2)
#     # c = torch.randperm(2)
#     # t = t[r[:, None], c]
#     # With view
#     idx = torch.randperm(t.nelement())
#     arr = []
#     for i in range(batch_size):
#         t = t.view(-1)[idx].view(t.size())
#         arr.append(t)
#     ret = torch.stack(arr)
#     return ret

#
# def train(model, optimizer, loss_function, train_loader, epoch, vector_size, channels, log_interval=100, cuda=True):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = tensor(data)
#         if cuda:
#             data = data.cuda()
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar, vector_size, channels)
#         loss.backward()
#         train_loss += loss.data[0]
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))
#
#
# def test(model, test_loader, epoch, vector_size, channels, cuda=True):
#     model.eval()
#     test_loss = 0
#     for i, (data, _) in enumerate(test_loader):
#         data = tensor(data, volatile=True)
#         if cuda:
#             data = data.cuda()
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar, vector_size, channels).data[0]
#
#     test_loss /= len(test_loader.dataset)
#     print('epoch: {}====> Test set loss: {:.4f}'.format(epoch, test_loss))
#


# from https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main():
    pass
