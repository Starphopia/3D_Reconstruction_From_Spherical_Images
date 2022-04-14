import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from distorter import Distorter


class DistortedConv2D(nn.Module):
    ''' Performs convolution with distorted sampling locations. '''
    def __init__(self, in_c, out_c, stride=1, mode="bilinear", kernel_size=3):
        ''' Initialises the weights '''
        super(DistortedConv2D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode
        self.kernel_size = kernel_size

        # Initialises the weights
        self.weights = Parameter(torch.Tensor(out_c, in_c, kernel_size, kernel_size))
        self.bias = Parameter(torch.Tensor(out_c))
        self.grid_shape = None
        self.grid = None

        self.clear_weights_biases()

    def clear_weights_biases(self):
        ''' Resets the weights to their default'''
        nn.init.xavier_uniform(self.weights)
        self.bias.data.zero_()

    def forward(self, input):
        IMAGE_HEIGHT = input.shape[2]
        IMAGE_WIDTH = input.shape[3]
        if self.grid_shape is None or self.grid_shape != tuple(input.shape[2:4]):
            distorter = Distorter(IMAGE_WIDTH, IMAGE_HEIGHT, self.kernel_size)
            self.grid_shape = tuple(input.shape[2:4])
            coordinates = distorter.distort_all_points()
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(input.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(input.shape[0], 1, 1, 1)

        sampled = nn.functional.grid_sample(input, grid, mode=self.mode)
        output = nn.functional.conv2d(sampled, self.weight, self.bias, stride=self.kernel_size)
        return output

    #TODO test the convolution