import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class DistortionAwareMaxPool(nn.MaxPool2d):
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), padding=0, dilation=1,
               return_indices = False, ceil_mode = False):
        super(DistortionAwareMaxPool, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                                                  dilation=dilation, return_indices=return_indices,
                                                  ceil_mode=ceil_mode)
        if isinstance(kernel_size, int):
            self.KH, self.KW = (kernel_size, kernel_size)
        else:
            self.KH, self.KW2 = kernel_size
        self.sampling_grid = None
        self.H = None
        self.W = None
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
    def forward(self, input):
        # Remembers the distorted sampling locations for the last iamge size.
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
            
        input = nn.functional.grid_sample(input, self.sampling_grid, align_corners=True, mode="bilinear")
        input = nn.functional.max_pool2d(input, kernel_size=(self.KH, self.KW), stride=self.kernel_size)
        
        return input