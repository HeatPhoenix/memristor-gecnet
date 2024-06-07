import torch
from torch import nn

class Multiply(nn.Module):
  def __init__(self, scalar):
    super(Multiply, self).__init__()
    self.scalar = scalar

  def forward(self, tensors):
    result = tensors * self.scalar
    return result