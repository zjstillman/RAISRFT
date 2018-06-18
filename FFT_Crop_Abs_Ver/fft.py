import numpy as np

def fftc(input, axes=None):
    dtype = input.dtype

    input = np.fft.ifftshift(input, axes=axes)
    output = np.fft.fftn(input, axes=axes, norm='ortho')
    output = np.fft.fftshift(output, axes=axes)

    return output.astype(input.dtype)


def ifftc(input, axes=None):
    dtype = input.dtype

    input = np.fft.ifftshift(input, axes=axes)
    output = np.fft.ifftn(input, axes=axes, norm='ortho')
    output = np.fft.fftshift(output, axes=axes)

    return output.astype(input.dtype)
