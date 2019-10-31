"""
Implementation of Large Memory with Product Keys as in https://arxiv.org/abs/1907.05242
"""
import torch
import torch.nn as nn


class SubKeyColumn(nn.Module):
    """
    Large Memory Product Key pre-processing network
    divides the input tensor in two smaller networks, outputs 2 results in the processing order
    """
    def __init__(self, indim, qshape, kshape, V):
        """

        :param indim:
        :param qshape:
        :param kshape:
        :param V: the _shared_ values tensor for the output
        """
        assert indim % 2 == 0
        # pre-processing network
        # q : x -> q(x) belongs to Real domain of dimension dq (R^dq)
        lin1 = nn.Linear(indim, indim//2)  # , bias=False)
        bn = nn.BatchNorm1d()
        self.preprocess = nn.ModuleList([lin1, bn])

    def forward(self, x):
        x1 = self.preprocess(x)
        # q =
        pass


class PKPreprocess(nn.Module):
    """
    Large Memory Product Key pre-processing network
    divides the input tensor in two smaller networks, outputs 2 results in the processing order
    the 2 tensors correspond to the
    """
    def __init__(self, indim):
        """

        :param indim: input dimension
        """
        pass


# def product_key_set(c1,c2,dim1=-2, dim2=-1):
def product_key_set(c1, c2):
    """
    Creates the product key set of two tensors as in https://arxiv.org/abs/1907.05242
    The result is the outer product with respect to the vector concatenation operator of 2 codebooks c1 and c2
    Ideas come from discussion here:
    https://discuss.pytorch.org/t/how-to-create-a-combination-of-concatenations-with-two-tensors/28709/2
    ONLY works on the last 2 dimensions of the codebooks, both codebooks must be the same size

    :param c1: first codebook set where dim = N
    :param c2: second codebook set where dim = M
    :return : new tensor dimension NxM (all the other dimensions stay the same)
    """
    dim1, dim2 = -2, -1  # I fix them to avoid issues
    assert c1.dim() == c2.dim()
    #     assert len(c1.shape) > dim1 >= -len(c1.shape)
    #     assert len(c2.shape) > dim2 >= -len(c2.shape)
    # add the new dimension where cat will work
    #     print(c1.shape,c2.shape)
    X1 = c1.unsqueeze(dim1)
    Y1 = c2.unsqueeze(dim2)
    #     print(X1.shape,Y1.shape)
    # setting dimensions to work on
    s1 = [1] * X1.dim()
    s1[dim1] = c1.shape[dim1]
    s2 = [1] * Y1.dim()
    s2[dim2] = c2.shape[dim1]
    #     print(s1,s2)
    X2 = X1.repeat(*s1)
    Y2 = Y1.repeat(*s2)
    #     print(X1.shape,X2.shape)
    s3 = list(c1.shape)  # [-1] * c1.dim()
    s3[dim2] = c1.shape[dim2] + c2.shape[dim2]  # == Z.shape[dim2]  # ==
    s3[dim1] = c1.shape[dim1] + c2.shape[dim1]
    if len(s3) > 2:
        s3[dim1 - 1] = -1
    #     print(s3)
    Z = torch.cat([X2, Y2], dim2)
    #     print(Z.shape)
    #     Z = Z.reshape(-1,Z.shape[-1])  #here an error
    Z = Z.reshape(*s3)
    #     print(Z.shape)
    return Z
