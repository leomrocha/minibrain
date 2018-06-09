import numpy as np
import sympy
from sympy import abc
from sympy import lambdify
from sympy import fourier_series, pi


################################################################################
## Fourier Series Approximations
################################################################################

def get_fourier_approx_function(min_val=0., max_val=1., f=abc.x, terms=20, lib=numpy):
    """
    Returns a fourier approximation for the given function
    @param min_val: minimum value that needs to be approximated
    @param max_value: maximum value that needs to be approximated
    @param f: the given function, linear by default. it MUST be dependent on only one value abc.x
    @param terms: the number of Fourier terms on the approximation
    @param lib: the library used behind the scenes for the returned function
                possible values: [math|numpy|mpmath]

    @return a function using the given library as a base implementation.
    """
    fourier = fourier_series(f, (f, min_val, max_val)).truncate(terms)
    f_res = lambdify(abc.x, f, lib)
    return f_res


class Lin2Fourier():
    """
    Fourier Series Inspired Approximation of a linear function encoding.
    Takes the linear function as one period of a SawTooth-Wave and encodes
    the period with enough elements to approximate the function wiht the
    resolution needed for a (close to) perfect reconstruction of the input
    """
    def __init__(self, min_val, max_val, terms=20, neg_allowed=True):
        """
        @param min_val: minimum value to encode
        @param max_val: maximum value to encode
        @param terms: number of terms that will be used during encoding
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
        fs = fourier_series(abc.x, (abc.x, min_val, max_val)).truncate(terms)
        fs_args = fs.args  # Fourier Series Arguments
        #create a pairwise (coefficient, formula)
        coef_fact = [ lambdify(abc.x, (np.product(f.args[:-1]), "numpy")(0),  # evaluate  sympy.pi to numeric
                      lambdify(abc.x, f.args[-1], "numpy")  ) for f in fs_args  # create lambda functions
                     ]
        self.coefficients, self.factors = zip(*coef_fact)
        self.coefficients = np.stack(self.coefficients)

    def encode(self, x):
        """
        @param x: input vector to encode
        @return vector encoding the input x in the value points of x for the sinusoidal encoders
        """
        vec = [f(x) for f in self.factors]
        ret = np.stack(vec)
        if not self.neg_allowed:
            ret = (ret + 1. ) / 2.  # pull all the encoded values in range [0,1]
        return ret

    def decode(self, vec):
        """
        @param vec: multi-dimensional input vector to decode into the original signal
        """
        scale_factor = self.coefficients
        tx = vec * scale_factor[:,None]
        x = np.sum(tx, axis=0)
        if not self.neg_allowed:  # here as is commutative to make less computations
            x = (x * 2.) -1 # go back to the original range
        return x


################################################################################
## Schannon + Fourier inispired approximation
################################################################################
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
        self.periods = [2**i for i in range(self.n_low, self.n_high)]  # choose periods acording to sampling -> maybe we should add 3rds and 5ths of each to allow for more complex time pattern compositions
        # Fourier divisor coefficients of the Series
        self.coefficients = [2*i for i in range(self.n_low, self.n_high)]  # just counting the elements
        print(self.coefficients, self.periods)


    def encode(self, x):
        """
        @param x: input vector to encode
        @return vector encoding the input x in the value points of x for the sinusoidal encoders
        """
        vec = []
        for n,T in zip(self.coefficients, self.periods):
            #val = np.stack( np.sin(n * x / T) ) # base term of a SawTooth Wave of period T Fourier Series
            val = np.stack( np.sin( x / T) ) # it seems to work better for the mix without the constant
            vec.append(val)
        ret = np.stack(vec)
        if not self.neg_allowed:
            ret = (ret + 1. ) / 2.  # pull all the encoded values in range [0,1]
        return ret

    def decode(self, vec):
        """
        @param vec: multi-dimensional input vector to decode into the original signal
        """
        x = 0
        scale_factor = []
        for n,T in zip(self.coefficients, self.periods):
            #scale_factor.append((1./n)*T)  # Scale factor of each term
            scale_factor.append(T)  # Scale factor of each term
        scale_factor = np.array(scale_factor)
        tx = vec
        tx = tx * scale_factor[:,None]
        x = np.sum(tx, axis=0)
        if not self.neg_allowed:  # here as is commutative to make less computations
            x = (x * 2.) -1 # go back to the original range
        return x
