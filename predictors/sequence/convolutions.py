import numpy as np
import torch
#import pytorch_fft.fft.autograd as fft
import pytorch_fft.fft as fft

def fftshift(x):
    """
    One dimension vector fftshift
    """
    return fft.roll_n(x,0,int(x.shape[0]/2))

#One dimension vector ifftshift
ifftshift = fftshift

def complex_multiply(x,y):
    """
    Element wise multiplication for complex numbers represented as pairs
    """
    x_re, x_im = x
    y_re, y_im = y
    z_re = (x_re * y_re) - (x_im * y_im)
    z_im = (x_re * y_im) + (x_im * y_re)
    return (z_re,z_im)

def complex_divide(x,y):
    """
    Element wise division for complex numbers represented as pairs
    """
    x_re, x_im = x
    y_re, y_im = y #denominator
    num_re, num_im = complex_multiply(x,(y_re, -1*y_im)) #by complex conjugate
    den = (y_re * y_re) - (y_im * (-1*y_im)) # is + because of the conjugate operation
    res = (num_re / den ), (num_im / den)
    return res

def convolve(x,y):
    """
    One dimensional vector convolution
    """
    x_re, x_im = x
    y_re, y_im = y
    xtmp = fft.fft(x_re,x_im)
    x_fft = fftshift(xtmp[0]), fftshift(xtmp[1])
    ytmp = fft.fft(y_re, y_im)
    y_fft = fftshift(ytmp[0]), fftshift(ytmp[1])
    tmp_re1, tmp_im1 = complex_multiply(x_fft, y_fft)
    tmp_re2, tmp_im2 = ifftshift(tmp_re1), ifftshift(tmp_im1)
    tmp_re3, tmp_im3 = fft.ifft(tmp_re2, tmp_im2)
    return fftshift(tmp_re3), fftshift(tmp_im3)

def deconvolve(z, y):
    """
    One dimensional vector deconvolution
    """
    z_re, z_im = z
    y_re, y_im = y
    ztmp = fft.fft(z_re, z_im)
    z_fft = fftshift(ztmp[0]), fftshift(ztmp[1])
    ytmp = fft.fft(y_re, y_im)
    y_fft = fftshift(ytmp[0]), fftshift(ytmp[1])
    tmp_re1, tmp_im1 = complex_divide(z_fft, y_fft)
    tmp_re2, tmp_im2 = ifftshift(tmp_re1), ifftshift(tmp_im1)
    tmp_re3, tmp_im3 = fft.ifft(tmp_re2, tmp_im2)
    return fftshift(tmp_re3), fftshift(tmp_im3)
