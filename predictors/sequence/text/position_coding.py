import numpy as np

FibArray = [0, 1]


def fib(n):
    if n < 0:
        print("Incorrect input")
    elif n <= len(FibArray):
        return FibArray[n - 1]
    else:
        temp_fib = fib(n - 1) + fib(n - 2)
        FibArray.append(temp_fib)
        return temp_fib


def get_coord_emb(shape=(1024, 22), fibinit=6):
    """
    Computes #channels coordinates for a vector of the given length.
    The coordinates are computed as follow,:
        if fibinit > 0 uses shape[1] elements from fibonacci series starting from fibinit in the series
         and computes the sine & cosine for the #fibonacci values in 0->2*PI
        else computes the sine & cosine the 0->2*PI range for each value in 1->shape[1]
    @param shape: shape (length,channels) of the embedding vector
    @param fibinit: if 0 uses linear, if >0 uses fibonacci series
    @return: a vector of shape of the input value
    """

    assert (len(shape) == 2 and shape[0] > 100 and shape[1] > 0)
    ncoords = shape[1] // 2
    d_coord = shape[0]
    # get steps
    if fibinit > 0:
        # Fibonacci numbers so the signals can mix and give longer relations and have absolute like ordering which can
        # be used for longer sentences than the given input
        fib(ncoords + fibinit)
        steps = FibArray[fibinit:ncoords + fibinit]
    else:
        # Linear relations so the signals are more time independent and there is only relative ordering into the
        # input vector only
        steps = [d_coord // (i + 1) for i in range(ncoords)]
    PI2 = 2 * np.pi

    ret = []
    for stp in steps:
        arr = np.arange(0, PI2, PI2 / float(stp))
        oarr = np.tile(arr, int(np.ceil(float(d_coord) / stp)))
        ret.append(oarr[:d_coord])

    sret = np.stack([np.sin(ret), np.cos(ret)])
    return sret
