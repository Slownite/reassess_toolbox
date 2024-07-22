from torch import nn
import torch
from .I3D import Unit3D
class I3D_head(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.model = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )
    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor)->torch.Tensor:
        return self.model(X_1)
