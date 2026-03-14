import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1
        self.max_indices = np.zeros((batch_size, in_channels, output_height, output_width, 2), dtype=int)



        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                A_slice = A[:, :, i:i+self.kernel, j:j+self.kernel]
                A_slice_reshaped = A_slice.reshape(batch_size, in_channels, -1) 
                max_flat_idx = np.argmax(A_slice_reshaped, axis=2) 

                Z[:, :, i, j] = np.max(A_slice_reshaped, axis=2)

                max_row, max_col = np.unravel_index(max_flat_idx, (self.kernel, self.kernel))
                self.max_indices[:, :, i, j, 0] = max_row
                self.max_indices[:, :, i, j, 1] = max_col

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        batch_size, in_channels, input_height, input_width = self.A.shape
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1
        batch_size, out_channels, output_height, output_width = dLdZ.shape

        dLdA = np.zeros_like(self.A)
        for i in range(output_height):
            for j in range(output_width):
                for c in range(in_channels):
                    max_row = self.max_indices[:, c, i, j, 0]
                    max_col = self.max_indices[:, c, i, j, 1]
                    dLdA[np.arange(batch_size), c, i+max_row, j+max_col] += dLdZ[:, c, i, j]
    
        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape

        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                A_slice = A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j] = np.mean(A_slice, axis=(2,3))

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, input_height, input_width = self.A.shape
        batch_size, out_channels, output_height, output_width = dLdZ.shape

        dLdA = np.zeros_like(self.A)
        for i in range(output_height):
            for j in range(output_width):
                for b in range(batch_size):
                    for c in range(in_channels):
                        dLdA[b, c, i:i+self.kernel, j:j+self.kernel] += dLdZ[b, c, i, j] / (self.kernel ** 2)
        
        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)  

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)  

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)
        
        return dLdA
