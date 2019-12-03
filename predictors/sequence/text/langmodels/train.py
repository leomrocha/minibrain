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
from .models import *

# TODO put this in a config file
fcodebook = "/home/leo/projects/minibrain/predictors/sequence/text/utf8-codes/utf8_codebook_overfit_matrix_2seg_dim64.npy"
dataset_train = "/home/leo/projects/Datasets/text/UniversalDependencies/ud-treebanks-v2.4/traindev_np_batches_779000x3x1024_uint16.npy"
BASE_DATA_DIR_UD_TREEBANK = "/home/leo/projects/Datasets/text/UniversalDependencies/ud-treebanks-v2.4"


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_test(model, checkpoint_path, base_name, test=True, test_accuracy=False, max_data=45):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_train = np.load(dataset_train)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters())
    # loss_function = F.nll_loss
    loss_function = pos_loss_function
    epoch_size = 10000
    batch_size = 50
    # TODO tis is for testing purposes
    # data = data_train
    data = data_train[-1000 * batch_size:, :, :]  # just for the trials, use the last 1000 batches only
    test_data = None
    if test:
        test_data = load_test_data(BASE_DATA_DIR_UD_TREEBANK)
    epochs = chunks(data, epoch_size, dim=0)
    epoch_count = 1
    for e in epochs:
        batches = chunks(e, batch_size, dim=0)
        train(model, optimizer, loss_function, batches, epoch_count, epoch_size, device)
        torch.cuda.empty_cache()
        # checkpoint
        cid = f"{epoch_count:04}"  # cid = str(epoch_count).zfill(4)
        model.network.save_checkpoint(checkpoint_path, base_name, cid)
        # TODO test loss and accuracy to be measured in CPU (or another GPU)
        # with batches bigger than 50 my GPU is out of memory
        test(model, loss_function, test_data, epoch_count, device, max_data)
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
    model.network.load_checkpoint(checkpoint_path)
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
    model.network.load_checkpoint(checkpoint_path)
    test_data = load_test_data(test_data_path)
    print("launching Accuracy test in CPU")
    test_accuracy(model, test_data, epoch_count, device, max_data)


def err_ckb(err):
    print("error with the subprocess ", err)


def train_cputest(model, checkpoint_path, base_name, test_accuracy=True, max_data=45):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        fchkpoint = model.network.save_checkpoint(checkpoint_path, base_name, cid)
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


def main_conv1d():
    utf8codes = np.load(fcodebook)
    utf8codes = utf8codes.reshape(1987, 64)
    model = OLD_Conv1DPoS(utf8codes)
    path = "./trained_models/conv1dcol"
    base_name = "conv1dcol_nll-loss"
    train_test(model, path, base_name)


def main_gatedconv1d():
    utf8codes = np.load(fcodebook)
    utf8codes = utf8codes.reshape(1987, 64)
    model = GatedConv1DPoS(utf8codes)
    path = "./trained_models/GatedConv1DCol"
    base_name = "GatedConv1DPoS_nll-loss"
    train_test(model, path, base_name)
