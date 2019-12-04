from datetime import datetime
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .utils.preprocess_conllu import *
from .utils.helpers import *
from .utils.tools import *
from .models import *

# TODO put this in a config file
fcodebook = "/home/leo/projects/minibrain/predictors/sequence/text/utf8-codes/utf8_codebook_overfit_matrix_2seg_dim64.npy"
utf8codematrix = "/home/leo/projects/minibrain/predictors/sequence/text/utf8-codes/utf8_code_matrix_2seg.npy"
dataset_train = "/home/leo/projects/Datasets/text/UniversalDependencies/ud-treebanks-v2.4/traindev_np_batches_779000x3x1024_uint16.npy"
BASE_DATA_DIR_UD_TREEBANK = "/home/leo/projects/Datasets/text/UniversalDependencies/ud-treebanks-v2.4"

# cuda seems to reverse the GPU ids with CUDA id so ... mess
# Cuda maps cuda:0 to my RTX 2080ti (GPU#1) and
# Cuda maps cuda:1 to my GTX 1080 (GPU#0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train_test(model, checkpoint_path, base_name, max_seq_len=384, test_loss=True, test_accuracy=False, max_data=45):

    model = model.to(device)
    data_train = np.load(dataset_train)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters())
    # loss_function = F.nll_loss
    loss_function = pos_loss_function
    epoch_size = 10000
    batch_size = 50
    # TODO tis is for testing purposes
    data = data_train
    # data = data_train[-1000 * batch_size:, :, :]  # just for the trials, use the last 1000 batches only
    test_data = None
    if test_loss:
        test_data = load_test_data(BASE_DATA_DIR_UD_TREEBANK, max_data)
    epochs = chunks(data, epoch_size, dim=0)
    epoch_count = 1
    for e in epochs:
        batches = chunks(e, batch_size, dim=0)
        train(model, optimizer, loss_function, batches, epoch_count, epoch_size, device, max_seq_len)
        torch.cuda.empty_cache()
        # checkpoint
        cid = f"{epoch_count:04}"  # cid = str(epoch_count).zfill(4)
        model.save_checkpoint(checkpoint_path, base_name, cid)
        # TODO test loss and accuracy to be measured in CPU (or another GPU)
        # with batches bigger than 50 my GPU is out of memory
        if test_loss:
            test(model, loss_function, test_data, epoch_count, device, max_data, max_seq_len)
            torch.cuda.empty_cache()
        if test_accuracy:
            test_accuracy(model, test_data, epoch_count, device, max_data)
            torch.cuda.empty_cache()
        epoch_count += 1

    # model.network.save_model("./trained_models/conv1dcol", "conv1dcol_nll-loss_epoch-{}".format(epoch_count))


def test_async(checkpoint_path, test_data_path, epoch_count, device, max_data, test_acc=False):
    # load checkpoint
    # model is hardcoded for the moment
    utf8codes = np.load(fcodebook)
    utf8codes = utf8codes.reshape(1987, 64)
    model = GatedConv1DPoS(utf8codes).to(device)
    model.load_checkpoint(checkpoint_path)
    test_data = load_test_data(test_data_path)
    print("launching test in CPU")
    test(model, pos_loss_function, test_data, epoch_count, device, max_data)
    if test_acc:
        print("launching Accuracy test in CPU")
        test_accuracy(model, test_data, epoch_count, device, max_data)


def test_acc_async(checkpoint_path, test_data_path, epoch_count, device, max_data):
    # load checkpoint
    # model is hardcoded for the moment
    utf8codes = np.load(fcodebook)
    utf8codes = utf8codes.reshape(1987, 64)
    model = GatedConv1DPoS(utf8codes).to(device)
    model.load_checkpoint(checkpoint_path)
    test_data = load_test_data(test_data_path)
    print("launching Accuracy test in CPU")
    test_accuracy(model, test_data, epoch_count, device, max_data)


def err_ckb(err):
    print("error with the subprocess ", err)


# Note this is TOO slow, GPU test is 30-50 times faster than in CPU, so CPU not useful for practical purposes
def train_cputest(model, checkpoint_path, base_name, test_accuracy=True, max_data=45):
    pool = Pool(cpu_count() - 2)
    model = model.to(device)
    data_train = np.load(dataset_train)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters())
    # loss_function = F.nll_loss
    loss_function = pos_loss_function
    epoch_size = 10000
    batch_size = 50
    # TODO this is for testing purposes
    data = data_train
    # data = data_train[-1000*batch_size:, :, :]  # just for the trials, use the last 1000 batches only

    epochs = chunks(data, epoch_size, dim=0)
    epoch_count = 1
    for e in epochs:
        batches = chunks(e, batch_size, dim=0)
        train(model, optimizer, loss_function, batches, epoch_count, epoch_size, device)
        torch.cuda.empty_cache()
        # checkpoint
        cid = f"{epoch_count:04}"  # cid = str(epoch_count).zfill(4)
        fchkpoint = model.save_checkpoint(checkpoint_path, base_name, cid)
        # test loss and accuracy to be measured in CPU (or another GPU)
        # with batches bigger than less than 50 my GPU is out of memory
        res_test = pool.apply_async(test_async,
                                    [fchkpoint, BASE_DATA_DIR_UD_TREEBANK, epoch_count, device,
                                     max_data],
                                    error_callback=err_ckb)
        if test_accuracy:
            res_acc = pool.apply_async(test_acc_async,
                                       [fchkpoint, BASE_DATA_DIR_UD_TREEBANK, epoch_count, device,
                                        max_data],
                                       error_callback=err_ckb)
        torch.cuda.empty_cache()
        epoch_count += 1

    # model.network.save_model("./trained_models/conv1dcol", "conv1dcol_nll-loss_epoch-{}".format(epoch_count))


def old_main_conv1d():
    utf8codes = np.load(fcodebook)
    utf8codes = utf8codes.reshape(1987, 64)
    model = OLD_Conv1DPoS(utf8codes)
    path = "./trained_models/conv1dcol"
    base_name = "conv1dcol_nll-loss"
    train_test(model, path, base_name)


def old_main_gatedconv1d():
    utf8codes = np.load(fcodebook)
    utf8codes = utf8codes.reshape(1987, 64)
    model = GatedConv1DPoS(utf8codes)
    path = "./trained_models/GatedConv1DCol"
    base_name = "GatedConv1DPoS_nll-loss"
    train_test(model, path, base_name)


def main_conv1dcolnet():
    utf8codes = np.load(utf8codematrix)
    # utf8codes = utf8codes.reshape(1987, 324)
    encoder = Conv1DColNet(transpose_output=True)  # use default parameters
    decoder = LinearUposDeprelDecoder(transpose_input=False)
    model = NetContainer(utf8codes, encoder, decoder)
    path = "./trained_models/Conv1dColNet_try3"
    base_name = "Conv1dColNet_nll-loss"
    train_test(model, path, base_name)


CONV1D_PRETRAIN_FILE = "/home/leo/projects/minibrain/predictors/sequence/text/trained_models/Conv1dColNet/Conv1dColNet_nll-loss_0078.state_dict.pth"


def main_convattnet(conv1d_pretrain_file=CONV1D_PRETRAIN_FILE):
    utf8codes = np.load(utf8codematrix)
    # utf8codes = utf8codes.reshape(1987, 324)
    # the convolutional encoder must NOT be retrained (that is what I'm trying to test)
    # with torch.no_grad():
    #     conv1d_encoder = Conv1DColNet(transpose_output=False)  # use default parameters
    #     conv1d_decoder = LinearUposDeprelDecoder(transpose_input=False)
    #     conv1d_model = NetContainer(utf8codes, conv1d_encoder, conv1d_decoder)
    #     # load pre-trained conv1dcolnet
    #     # conv1d_model.load_checkpoint(conv1d_pretrain_file)
    #     # cleanup things that we'll not use, we just need the encoder
    #     del conv1d_model
    #     del conv1d_decoder
    #     torch.cuda.empty_cache()
    conv1d_encoder = Conv1DColNet(transpose_output=False)  # use default parameters
    encoder = ConvAttColNet(conv1d_encoder, transpose_output=False)
    decoder = LinearUposDeprelDecoder(transpose_input=False)
    model = NetContainer(utf8codes, encoder, decoder)
    print("Starting training for model with column type ConvAttNetCol and pretrained Conv1dColNet")
    print("Parameter model details: ")
    print("conv1d_encoder parameters {} from which {} are trainable ".
          format(count_parameters(conv1d_encoder), count_parameters(conv1d_encoder)))
    print("ConvAttColNet parameters {} from which {} are trainable ".
          format(count_parameters(encoder), count_parameters(encoder)))
    print("decoder parameters {} from which {} are trainable ".
          format(count_parameters(decoder), count_parameters(decoder)))
    print("Total model parameters {} from which {} are trainable ".
          format(count_parameters(model), count_parameters(model)))
    path = "./trained_models/ConvAttNet"
    base_name = "ConvAttNet_nll-loss"
    train_test(model, path, base_name, max_seq_len=384, max_data=60)
