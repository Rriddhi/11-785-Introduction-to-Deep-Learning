import numpy as np


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        k = self.upsampling_factor

        batch_size, in_channels, input_width = A.shape
        output_width = k * (input_width - 1) + 1

        Z = np.zeros((batch_size, in_channels, output_width))
        for i in range(input_width):
            Z[:, :, i*k] = A[:, :, i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        k = self.upsampling_factor

        batch_size, in_channels, output_width = dLdZ.shape
        input_width = (output_width - 1) // k + 1
        dLdA = np.zeros((batch_size, in_channels, input_width))
        for i in range(input_width):
            dLdA[:, :, i] = dLdZ[:, :, i*k]

        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        k = self.downsampling_factor
        
        self.batch_size, self.in_channels, self.input_width = A.shape
        self.output_width = (self.input_width - 1) // k + 1

        Z = np.zeros((self.batch_size, self.in_channels, self.output_width))
        for i in range(self.output_width):
            Z[:, :, i] = A[:, :, i*k]
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        k = self.downsampling_factor

        self.batch_size, self.in_channels, self.output_width = dLdZ.shape
        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_width))
        for i in range(self.output_width):
            dLdA[:, :, i*k] += dLdZ[:, :, i]

        return dLdA


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        k = self.upsampling_factor
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = k * (input_height - 1) + 1
        output_width = k * (input_width - 1) + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        for i in range(input_height):
            for j in range(input_width):
                Z[:, :, i*k, j*k] = A[:, :, i, j]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        k = self.upsampling_factor
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        input_height = (output_height - 1) // k + 1
        input_width = (output_width - 1) // k + 1
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        for i in range(input_height):
            for j in range(input_width):
                dLdA[:, :, i, j] = dLdZ[:, :, i*k, j*k]
        
        return dLdA


class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        k = self.downsampling_factor
        self.batch_size, self.in_channels, self.input_height, self.input_width = A.shape
        self.output_height = (self.input_height - 1) // k + 1
        self.output_width = (self.input_width - 1) // k + 1
        Z = np.zeros((self.batch_size, self.in_channels, self.output_height, self.output_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                Z[:, :, i, j] = A[:, :, i*k, j*k]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        k = self.downsampling_factor
        self.batch_size, self.in_channels, self.output_height, self.output_width = dLdZ.shape

        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_height, self.input_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                dLdA[:, :, i*k, j*k] += dLdZ[:, :, i, j]

        return dLdA
