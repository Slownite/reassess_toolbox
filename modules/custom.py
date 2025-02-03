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
    def __init__(self, input_dim=8192, hidden_dims=[4096, 2048, 1024, 512], num_classes=2, dropout_prob=0.3):
        super(X3D_head, self).__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())  # Non-linear activation
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch Normalization
            # Dropout for regularization
            layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes, bias=True))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

    def __str__(self):
        return "X3D_rgb"


class LSTMHead(nn.Module):
    def __init__(self, input_dim=8192, hidden_dim=512, num_layers=2, num_classes=2, dropout_prob=0.3):
        """
        LSTM-based model with a final fully connected layer for classification.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(LSTMHead, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # Dropout only for multi-layer LSTM
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes),
        )

    def forward(self, X):
        """
        Forward pass for the model.

        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output logits of shape (batch_size, num_classes).
        """
        # LSTM outputs
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(X)

        # Use only the last hidden state for classification
        # shape: (batch_size, hidden_dim)
        last_hidden_state = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc(last_hidden_state)
        return out

    def __str__(self):
        return "LSTMHead"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ProjectedTransformer(nn.Module):
    def __init__(self, input_dim=8192, d_model=512, nhead=8, num_layers=6, num_classes=2, seq_len=10):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU()
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        assert x.ndim == 3, f"(batch_size, seq_len, input_dim), got {x.shape}"

        x = self.projection(x)
        x = self.input_norm(x)
        x = self.positional_encoding(x)  # Add positional encoding

        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)

        # Instead of x[:, 0, :] (which assumes a CLS token), use mean pooling
        cls_token_state = x.mean(dim=1)

        logits = self.fc_out(cls_token_state)
        return logits  # No sigmoid if using CrossEntropyLoss
