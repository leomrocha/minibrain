import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F


###############
# from HuggingFace https://github.com/huggingface/transformers BERT implementation
def gelu_old(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
# end from HuggingFace
###############

###############
# Counting number of parameters
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###############


def get_activation_fn(activation):
    if activation == "sigmoid":
        return F.sigmoid
    elif activation == "tanh":
        return F.tanh
    elif activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        return None
        # raise RuntimeError("activation should be sigmoid/tanh/relu/gelu, not %s." % activation)
