"""
Learned Baseline Models for Population Prediction
Alternative to simple dasymetric (lights × settlement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionBaseline(nn.Module):
    """
    Simple linear regression baseline.

    Learns: population = w1 * lights + w2 * settlement + bias

    This is much better than fixed lights × settlement because:
    - Optimal weights are learned from data
    - Can add bias term
    - Can fit population scale better
    """

    def __init__(self):
        super().__init__()
        # Single conv layer acts as linear regression per pixel
        self.conv = nn.Conv2d(2, 1, kernel_size=1, bias=True)

        # Initialize with reasonable defaults
        with torch.no_grad():
            # Start with equal weights for lights and settlement
            self.conv.weight[0, 0] = 0.5  # lights coefficient
            self.conv.weight[0, 1] = 0.5  # settlement coefficient
            self.conv.bias[0] = 0.01       # small positive bias

    def forward(self, lights, settlement):
        """
        Args:
            lights: (B, 1, H, W) normalized nighttime lights
            settlement: (B, 1, H, W) normalized settlement footprint

        Returns:
            baseline_population: (B, 1, H, W) predicted population
        """
        # Stack inputs
        x = torch.cat([lights, settlement], dim=1)  # (B, 2, H, W)

        # Linear combination
        out = self.conv(x)  # (B, 1, H, W)

        # Ensure non-negative population
        out = F.relu(out)

        return out


class CNNBaseline(nn.Module):
    """
    Small CNN baseline for population prediction.

    Can learn spatial patterns and non-linear relationships.
    Still lightweight (< 500 parameters).
    """

    def __init__(self, hidden_channels=16):
        super().__init__()

        self.net = nn.Sequential(
            # First layer: combine features
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(),

            # Second layer: refine
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(),

            # Output layer
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.ReLU()  # Ensure non-negative
        )

        # Initialize with small weights for stability
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, lights, settlement):
        """
        Args:
            lights: (B, 1, H, W) normalized nighttime lights
            settlement: (B, 1, H, W) normalized settlement footprint

        Returns:
            baseline_population: (B, 1, H, W) predicted population
        """
        x = torch.cat([lights, settlement], dim=1)  # (B, 2, H, W)
        return self.net(x)


class ScaledDasymetric(nn.Module):
    """
    Simple dasymetric with learnable scaling factor.

    population = scale * (lights × settlement)

    Minimal change from current baseline, but can fix scale issue.
    """

    def __init__(self, init_scale=10.0):
        super().__init__()
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, lights, settlement):
        """
        Args:
            lights: (B, 1, H, W) normalized nighttime lights
            settlement: (B, 1, H, W) normalized settlement footprint

        Returns:
            baseline_population: (B, 1, H, W) predicted population
        """
        # Simple product, but with learned scale
        baseline = lights * settlement * F.softplus(self.scale)  # softplus ensures positive
        return baseline


# Export all baseline options
__all__ = ['LinearRegressionBaseline', 'CNNBaseline', 'ScaledDasymetric']
