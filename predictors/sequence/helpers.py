import numpy as np
import torch


# Counting number of parameters
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##

def get_upper_pow2(M):
    i = 1
    n = 0
    while (i <= M):
        i = i << 1
        n+=1
    return n


def get_lower_pow2(m):
    i = 1
    n = -1
    while (i <= m):
        i = i << 1
        n+=1
    return n


class Lin2WaveEncoder():
    """
    Fourier Series Inspired Approximation of a linear function encoding.
    Takes the linear function as one period of a SawTooth-Wave and encodes
    the period with enough elements to approximate the function wiht the
    resolution needed for a (close to) perfect reconstruction of the input
    """
    def __init__(self, min_val, max_val, neg_allowed=True):
        """
        @param min_val: minimum value to encode
        @param max_val: maximum value to encode
        @param neg_allowed=True : allow negative values in the encoded vecto. If False
                        will move the sinusoids to the range [0,1]
        """
        self.min = min_val
        self.max = max_val
        self.neg_allowed = neg_allowed
        self.n_low = np.max(1, get_lower_pow2(min_val))
        self.n_high = get_upper_pow2(max_val)
        self.n = self.n_high - self.n_low
        # Each period indicates a resolution level, this allows for different (time) scales
        self.periods = [2**i for i in range(self.n_low, self.n_high)]
        # Fourier divisor coefficients of the Series
        self.coefficients = [2*i for i in range(self.n_low, self.n_high)]

    def encode(self, x):
        """
        @param x: input ValueError
        @return vector encoding the input x in the value points of x for the sinusoidal encoders
        """
        vec = []
        for n,T in zip(self.coefficients, self.periods):
            val = np.stack( np.sin(n * x / T) ) # base term of a SawTooth Wave of period T Fourier Series
            vec.append(val)
        ret = np.stack(vec)
        if not self.neg_allowed:
            ret = (ret + 1. ) / 2.  # pull all the encoded values in range [0,1]
        return ret

    def decode(self, vec):
        x = 0
        scale_factor = []
        for n,T in zip(self.coefficients, self.periods):
            scale_factor.append((1./n)*T)  # Scale factor of each term
        scale_factor = np.array(scale_factor)
        tx = vec
        tx = tx * scale_factor[:,None]
        x = np.sum(tx, axis=0)
        if not self.neg_allowed:  # here as is commutative to make less computations
            x = (x * 2.) -1  # go back to the original range
        return x
