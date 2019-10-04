import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import tqdm

from .column_models import *


def clip_global_norm(model, clip):
    norms = []
    total_norm = 0

    for p in model.parameters():
        norm = p.grad.data.norm()

        if norm > clip:
            p.grad.data.div_(max(norm, 1e-6) / clip)


def train(model, criterion, optimizer, data):

    pbar = tqdm.tqdm(zip(data["input"], data["label"]))

    for X_batch, Y_batch in pbar:
        X_tensor = torch.from_numpy(X_batch).cuda()
        Y_tensor = torch.from_numpy(Y_batch.astype(np.int)).cuda()
        Y_tensor = Y_tensor.view(-1)
        model.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        clip_global_norm(model, 0.25)
        optimizer.step()
        pbar.set_description('Loss: {:.3f}'.format(loss.item()))


def test(model, test_data, nll=False):
    pbar = tqdm.tqdm(zip(test_data['input'], test_data['label']))

    if nll:
        criterion = nn.NLLLoss(size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(size_average=False)

    nllloss = 0

    for X_batch, Y_batch in pbar:
        X_tensor = torch.from_numpy(X_batch).cuda()
        Y_tensor = torch.from_numpy(Y_batch.astype(np.int)).cuda()
        Y_tensor = Y_tensor.view(-1)
        output, hidden = model(X_tensor, Y_tensor, training=False)
        nllloss += criterion(output, Y_tensor).item()

    loss = nllloss / (len(test_data['input']) * 128 * 20)

    print('Perplexity:', np.exp(loss))

    return loss


def run_epochs(data, test_data, model, criterion, optimizer, n_epochs=5, nll=False):
    for epoch in range(n_epochs):
        train(model, criterion, optimizer, data)
        test(model, test_data, nll)


def main():
    with open('/home/leo/projects/Datasets/text8.train.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('/home/leo/projects/Datasets/text8.test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    model = MixedConvLinearColumns()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=0.1, lr_decay=1e-5, weight_decay=1e-5)
    n_epochs = 2
    nll = False
    run_epochs(data, test_data, model, criterion, optimizer, n_epochs, nll)


if __name__ == "__main__":
    main()


