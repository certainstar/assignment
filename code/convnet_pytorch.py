"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels, Guess this corresponds to 3 of the first layer input
      n_classes: number of classes of the classification problem, Guess this corresponds to 10 of the last layer output    
    TODO:
    Implement initialization of the network.
    """
    super(ConvNet, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), # batch normalization.Helps accelerate training speed and stability
        nn.ReLU(inplace=True) # inplace=True save memory
    )
    self.layer2 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
    )
    self.layer4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.layer5_a = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
    )
    self.layer5_b = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
    )
    self.layer6 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.layer7_a = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    )
    self.layer7_b = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    )
    self.layer8 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.layer9_a = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    )
    self.layer9_b = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    )
    self.layer10 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.fc = nn.Linear(512, n_classes)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5_a(x)
    x = self.layer5_b(x)
    x = self.layer6(x)
    x = self.layer7_a(x)
    x = self.layer7_b(x)
    x = self.layer8(x)
    x = self.layer9_a(x)
    x = self.layer9_b(x)
    x = self.layer10(x)
    x = x.view(x.size(0), -1) # tiling tensor
    out = self.fc(x)

    return out
