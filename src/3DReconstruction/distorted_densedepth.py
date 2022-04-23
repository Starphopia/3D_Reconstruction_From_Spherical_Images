from distorter_conv import DistortionAwareConv
from distorter_maxpool import DistortionAwareMaxPool
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Obtains the densenet169 neural network.
        self.base_model = models.densenet169(pretrained=True)
        self.base_model = self.distort_layers(self.base_model)
        
    
    def distort_layers(self, base_model):
        """
        Goes through each layer in the preloaded DenseNet network and replace them with distortion aware convolutions
        :param base_model: the densenet model which layers will be replaced.
        :type base_model: nn.Module
        """
        for i, (name, layer) in enumerate(base_model.features._modules.items()):
            if "conv" in name: 
                conv_weights = layer.weight
                base_model.features[i] = DistortionAwareConv(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                                             kernel_size=layer.kernel_size, stride=layer.stride, 
                                                             padding=layer.padding, padding_mode=layer.padding_mode, 
                                                             bias=layer.bias)
                base_model.features[i].weight = conv_weights

            if "pool" in name:
                base_model.features[i] = DistortionAwareMaxPool(kernel_size=layer.kernel_size, stride=layer.stride,
                                                                padding=layer.padding, dilation=layer.dilation, 
                                                                ceil_mode=layer.ceil_mode)

            if "denseblock" in name:
                base_model.features[i] = self.distort_dense_block(layer)
                
            if "transition" in name:
                base_model.features[i] = self.distort_transition(layer)
        return base_model 
        

    def distort_transition(self, transition):
        """
        Replaces the Conv2D layer in a transition block of a DenseNet network with a distortion aware one.
        :param transition: Transition block in the DenseNet network.
        :type transition: Transition
        """
        conv = transition.conv
        conv_weights = transition.conv.weight
        transition.conv = DistortionAwareConv(in_channels=conv.in_channels, out_channels=conv.out_channels,
                                              kernel_size=conv.kernel_size, stride=conv.stride,
                                              padding=conv.padding, padding_mode=conv.padding_mode,
                                              bias=conv.bias)
        transition.conv.weight = conv_weights
        return transition
    
   
    def distort_dense_block(self, denseblock):
        """
        Replaces the Conv2D and MaxPool2D layers in each Dense Layer within the DenseBlock.
        :param denseblock: Dense Block consisting of dense layers.
        :param denseblock: Dense Block
        """
        for name, layer in denseblock.items():
            denseblock.update({name : self.distort_dense_layer(layer)})
        return denseblock

    def distort_dense_layer(self, denselayer):
        """
        :param denselayer: the denselayer which convolution we are replacing.
        :type denselayer: Dense Layer
        """
        conv1 = denselayer.conv1
        conv1_weights = conv1.weight
        denselayer.conv1 = DistortionAwareConv(in_channels=conv1.in_channels, out_channels=conv1.out_channels,
                                               kernel_size=conv1.kernel_size, stride=conv1.stride, 
                                               padding=conv1.padding, padding_mode=conv1.padding_mode, 
                                               bias=conv1.bias)
        denselayer.conv1.weight = conv1_weights
        conv2 = denselayer.conv2
        conv2_weights = conv2.weight
        denselayer.conv2 = DistortionAwareConv(in_channels=conv2.in_channels, out_channels=conv2.out_channels,
                                               kernel_size=conv2.kernel_size, stride=conv2.stride,
                                               padding=conv2.padding, padding_mode=conv2.padding_mode,
                                               bias=conv2.bias)
        denselayer.conv2.weight = conv2_weights
        return denselayer

        
    def forward(self, x):
        features = [x]
        for key, value in self.base_model.features._modules.items():
            features.append(value(features[-1]))
            
        return features
        
        
        
class UpSample(nn.Module):
    def __init__(self, num_channels_in : int, num_channels_out : int):
        super(UpSample, self).__init__()
        
        self._layers = {
            "conv1" : DistortionAwareConv(num_channels_in, num_channels_out, kernel_size=(3, 3),
                                          stride=(1, 1), padding="same"),
            "leaky_relu" : nn.LeakyReLU(0.2),
            "conv2" : DistortionAwareConv(num_channels_out, num_channels_out, kernel_size=(3,3),
                                          stride=(1, 1), padding="same"),
        }
    
    def forward(self, x, concat_with):        
        x = F.interpolate(x, size=[2, 2], mode="bilinear", align_corners=True)
        x = torch.cat([x1, concat_with], dim=1)
        x = self._layers["conv1"](x)
        x = self._layers["leaky_relu"](x)
        x = self._layers["conv2"](x)
        return self._layers["leaky_relu"](x)
        
        
    def get_layers(self):
        return self._layers.copy()
        

class Decoder(nn.Module):
    def __init__(self, num_input_features=1664):
        super(Decoder, self).__init__()
        self._layers = {
            "conv2" : DistortionAwareConv(num_input_features, num_input_features, kernel_size=(1, 1),
                                          stride=(1, 1), padding=0),
            "up1" : UpSample(num_input_features, num_input_features // 2),
            "up2" : UpSample(num_input_features // 2, num_input_features // 4),
            "up3" : UpSample(num_input_features // 4, num_input_features // 8),
            "up4" : UpSample(num_input_features // 8, num_input_features // 16), 
            "conv3" : DistortionAwareConv(num_input_features // 16, 1, stride=(1, 1),
                                          kernel_size=(3, 3), padding="same")
        }

    def forward(self, x):
        x_block0, x_block1, x_block2, x_block3, x_block4 = x[3], x[4], x[6], x[8], [12]
        x_d0 = self._layers["conv2"](F.relu(x_block4))

        x_d1 = self._layers["up1"](x_d0, x_block3)
        x_d2 = self._layers["up2"](x_d1, x_block2)
        x_d3 = self._layers["up3"](x_d2, x_block1)
        x_d4 = self._layers["up4"](x_d3, x_block0)
        return self._layers["conv3"](x_d4)
    
        
        
class DistortedDenseDepthModel(nn.Module):
    """
    DenseDepth network but with the normal Conv2D and MaxPool2D layers replaced with 
    their distortion aware versions DistortionAwareConv and DistortionAwareMaxPool.
    """
    
    def __init__(self):
        super(DistortedDenseDepthModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )