from torch import nn
import torch
import torch.nn.functional as F
from .I3D import Unit3D
class I3D_head(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.adapt_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
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
    def __str__(self):
        return "I3D_rgb"
    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor)->torch.Tensor:
        x = self.adapt_pooling(X_1)
        x = self.model(x)
        logits = x.squeeze(3).squeeze(3)
        mean_logits =  logits.mean(2)
        return F.softmax(mean_logits, dim=1)
