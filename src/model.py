"""
2D Convolutional Neural Network (CNN) tailored for Audio Spectrograms.

Architecture highlights:
- Anisotropic receptive fields: preserves spectral harmonic patterns and temporal dynamics
- Squeeze-and-Excitation (SE) blocks: models inter-channel spectral correlations
- Residual convolutional blocks: prevents gradient degradation in deeper layers
- Global Pooling (Adaptive Average + Max Pooling): duration-invariant representation
- Regularization with Dropout and Batch Normalization
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel-wise Squeeze-and-Excitation block.

    Explicitly models channel dependencies, allowing the network to emphasize
    salient harmonic spectral features while suppressing background noise.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced_dim = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_dim, bias=False)
        self.fc2 = nn.Linear(reduced_dim, channels, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Freq, Time)
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling over F and T
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation
        y = self.activation(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Scale
        return x * y


class ConvBlock(nn.Module):
    """Convolutional Block with Residual connection, BatchNorm, Activation, and SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: Optional[Tuple[int, int]] = (2, 2),
        use_se: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection if channel dimensions change
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=pool_size) if pool_size else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual, inplace=True)

        out = self.pool(out)
        out = self.dropout(out)
        return out


class AudioSpectrogramCNN(nn.Module):
    """Deep 2D CNN architecture tailored for Log-Mel Spectrogram Event Classification.

    Input shape:  (Batch, 1, n_mels, time_steps)
    Output shape: (Batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Four stages of progressive feature abstraction
        # Stage 1: (base_channels) -> 32
        self.block1 = ConvBlock(base_channels, base_channels, pool_size=(2, 2), dropout=0.1)
        # Stage 2: 32 -> 64
        self.block2 = ConvBlock(base_channels, base_channels * 2, pool_size=(2, 2), dropout=0.1)
        # Stage 3: 64 -> 128
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4, pool_size=(2, 2), dropout=0.15)
        # Stage 4: 128 -> 256
        self.block4 = ConvBlock(base_channels * 4, base_channels * 8, pool_size=(2, 2), dropout=0.2)

        feature_dim = base_channels * 8

        # Dual Global Pooling: Average + Max Pooling concatenation for rich spectral descriptors
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classifier head
        classifier_input_dim = feature_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense fixed-length audio embeddings before classification projection.

        Parameters
        ----------
        x : torch.Tensor
            Spectrogram input tensor (Batch, 1, n_mels, time_steps).

        Returns
        -------
        torch.Tensor
            Feature embeddings of shape (Batch, 2 * feature_dim).
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        avg_pooled = self.global_avg_pool(x).flatten(1)
        max_pooled = self.global_max_pool(x).flatten(1)
        features = torch.cat([avg_pooled, max_pooled], dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass supporting (B, C, F, T), (C, F, T), or (F, T).

        Parameters
        ----------
        x : torch.Tensor
            Spectrogram input tensor.

        Returns
        -------
        torch.Tensor
            Logits of shape (Batch, num_classes).
        """
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            if x.size(0) == 1:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)

        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Loads state_dict with automatic adaptation if checkpoint class count differs."""
        for key in ("classifier.4.weight", "classifier.4.bias"):
            if key in state_dict:
                ckpt_classes = state_dict[key].shape[0]
                if ckpt_classes != self.num_classes:
                    last_linear = self.classifier[4]
                    in_features = last_linear.in_features
                    self.classifier[4] = nn.Linear(in_features, ckpt_classes)
                    self.num_classes = ckpt_classes
                break
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Alias for flexible integration
AudioCNN = AudioSpectrogramCNN

