import torch
import torch.nn as nn
# good that PyTorch v1.3.0+ has Transformers already implemented
from torch.nn.modules.transformer import Transformer
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F



class DynConvAttention(nn.Module):
