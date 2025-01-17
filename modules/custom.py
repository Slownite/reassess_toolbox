from torch import nn
import torch
import torch.nn.functional as F
from .I3D import Unit3D


class RGB_I3D_head(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.adapt_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=F.relu,
            use_batch_norm=True,
            use_bias=True,
            name="logits",
        )

    def __str__(self):
        return "I3D_rgb"

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor) -> torch.Tensor:
        x = self.adapt_pooling(X_1)
        x = self.model(x)
        logits = x.squeeze(3).squeeze(3)
        mean_logits = logits.mean(2)
        return mean_logits


class OF_I3D_head(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.adapt_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=F.relu,
            use_batch_norm=True,
            use_bias=True,
            name="logits",
        )

    def __str__(self):
        return "I3D_rgb"

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor) -> torch.Tensor:
        x = self.adapt_pooling(X_2)
        x = self.model(x)
        logits = x.squeeze(3).squeeze(3)
        mean_logits = logits.mean(2)
        return mean_logits


class X3D_head(nn.Module):
    def __init__(self, input_dim=401408, hidden_dims=[2048, 1024, 512], num_classes=2, dropout_prob=0.5):
        super(X3D_head, self).__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())  # Non-linear activation
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch Normalization
            # Dropout for regularization
            layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

    def __str__(self):
        return "X3D_rgb"
