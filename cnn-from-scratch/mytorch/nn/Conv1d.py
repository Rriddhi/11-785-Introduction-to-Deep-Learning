# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, input_width = A.shape[0], A.shape[2]

        output_width = input_width - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_width))

        for i in range(output_width):
            A_slice = A[:, :, i:i+self.kernel_size]
            Z[:, :, i] = np.tensordot(A_slice, self.W, axes=([1, 2], [1, 2])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_width = dLdZ.shape
        batch_size, in_channels, input_width = self.A.shape
        self.dLdW = np.zeros(self.W.shape)
        for i in range(self.kernel_size):
            A_slice = self.A[:, :, i:i+output_width]          
            self.dLdW[:, :, i] = np.tensordot(
                dLdZ, A_slice,
                axes=([0, 2], [0, 2])                        
            )

        self.dLdb = np.sum(dLdZ, axis=(0, 2)) # TODO

        W_flipped = np.flip(self.W, axis=2) # TODO

        dLdA = np.zeros((batch_size, in_channels, input_width))

        pad = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (pad, pad)))
        for i in range(input_width):
            dLdZ_slice = dLdZ_padded[:, :, i:i+self.kernel_size]
            dLdA[:, :, i] = np.tensordot(dLdZ_slice, W_flipped,axes=([1, 2], [0, 2]) )

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Pad the input appropriately using np.pad() function
        # TODO
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)))

        # Call Conv1d_stride1
        # TODO
        Z = self.conv1d_stride1.forward(A_padded)

        # downsample
        #TODO
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdZ_upsampled = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ_upsampled)

        # Unpad the gradient
        # TODO
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad]

        return dLdA
