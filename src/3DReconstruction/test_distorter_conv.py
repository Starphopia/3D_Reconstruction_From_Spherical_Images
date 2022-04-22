import urllib.request
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

# Code from 
class SphereConv2d(nn.Conv2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
    super(SphereConv2d, self).__init__(
      in_channels, out_channels, kernel_size,
      stride, padding, dilation, groups, bias, padding_mode)
    self.grid_shape = None
    self.grid = None
    print("HELLO")

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw)
    x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size)

    return x  # (B, out_c, H/stride_h, W/stride_w)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SphereMaxPool2d(nn.MaxPool2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, kernel_size=(3, 3), stride=1, padding=0, dilation=1,
               return_indices: bool = False, ceil_mode: bool = False):
    super(SphereMaxPool2d, self).__init__(
      kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    if isinstance(kernel_size, int):
      self.kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
      self.stride = (stride, stride)

    self.grid_shape = None
    self.grid = None
    print("WHY")

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)

    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=False, mode='bilinear')  # (B, in_c, H*Kh, W*Kw)

    x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.kernel_size)

    return x  # (B, out_c, H/stride_h, W/stride_w)


url = "https://media.istockphoto.com/photos/seamless-360-vr-home-office-panorama-3d-illustration-of-modern-picture-id1098049256?k=20&m=1098049256&s=612x612&w=0&h=inULdFulQcHOSmY1RRL_aEBVOq5h5OkTOxG6iM_Z2nA="
urllib.request.urlretrieve(url, "img.png")
  
img = Image.open("img.png")
data = asarray(img)
new_data = data.copy()

plt.imshow(new_data)

# Testing the Convolution Modules
cnn = SphereConv2D(3, 5, stride=3)
cnn = DistortionAwareConv(3, 5, kernel_size=(3, 3), stride=(3, 3))
out = cnn(torch.randn(2, 3, 10, 10))
print('SphereConv2d(3, 5, 1) output shape: ', out.size())

# Testing the MaxPool Modules
img = new_data / 255
plt.imsave('demo_original.png', img)
img = img.transpose([2, 0, 1])
img = np.expand_dims(img, 0)  # (B, C, H, W)

pool = SphereMaxPool2d(stride=(100, 100), kernel_size=(3, 3))
out = pool(torch.from_numpy(img).float())
out = np.squeeze(out.detach().numpy(), 0).transpose([1, 2, 0])
plt.imsave('expected_demo_pool_3x3.png', abs(out))
pool = DistortionAwarePool((3, 3), stride=(100, 100))
out2 = pool(torch.from_numpy(img).float())
out2 = np.squeeze(out2.detach().numpy(), 0).transpose([1, 2, 0])
plt.imsave('demo_pool_3x3.png', abs(out2))
