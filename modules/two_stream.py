import torch
from torch import nn
from torch.nn import functional as F

class TwoStreamNet(nn.Module):
  def __init__(self, num_classes: int, is_extractor: bool =False)->None:
    super().__init__()
    self.spatial_stream = nn.Sequential(
    nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=3),  # Input from images
    nn.BatchNorm2d(96),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),

    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),

    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
    )
    self.decision = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),

      nn.Linear(512, 2048),
      nn.ReLU(inplace=True),
      )

  def forward(self, X: torch.Tensor)->torch.Tensor:
    spatial = X[0]
    temporal = X[1]
    x = self.stream(x)
    x = self.decision(x)
    return x


def combined_output(spatial, temporal):
    output_combined = (spatial + temporal) / 2.0
    return output_combined
