import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class DistortionAwareConv(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 padding_mode="zeros", dilation=1, groups=1, bias=True):
        super(DistortionAwareConv, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride, 
                                                padding=padding, padding_mode=padding_mode,
                                                dilation=dilation, groups=groups, bias=bias)
        (self.KW, self.KH) = kernel_size
        self.sampling_grid = None
        self.H = None
        self.W = None
        self.stride = stride
        print(self.weight)
        
    def forward(self, input):
        # Remembers the distorted sampling locations for the last image size.
        if (self.H, self.W) != input.shape[2:4]:
            (self.H, self.W) = input.shape[2:4]
            
            # Computes the grid
            with torch.no_grad():
                self.sampling_grid = torch.FloatTensor(SamplingGridDistorter(self.W, self.H, self.KW, 
                                                       self.KH, self.stride).get_sampling_grid()).to(input.device)
                self.sampling_grid.requires_grad = False
                

        # Adjusts the grid according to the batch size.
        with torch.no_grad():
            self.sampling_grid = self.sampling_grid.repeat(input.shape[0], 1, 1, 1)
            self.sampling_grid.requires_grad = False
            
        input = nn.functional.grid_sample(input, self.sampling_grid, align_corners=True, mode="nearest")
        input = nn.functional.conv2d(input, self.weight, self.bias, stride=self.stride)
        
        return input