"""
models/attention.py

Self-Attention module for Hybrid VAE-AttentionGAN-GMM.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Self-Attention Layer
    """

    def __init__(self, in_channels):
        super().__init__()

        self.query = nn.Conv2d(
            in_channels,
            in_channels // 8,
            kernel_size=1
        )

        self.key = nn.Conv2d(
            in_channels,
            in_channels // 8,
            kernel_size=1
        )

        self.value = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        batch_size, channels, height, width = x.size()

        query = self.query(x).view(
            batch_size,
            -1,
            height * width
        ).permute(0, 2, 1)

        key = self.key(x).view(
            batch_size,
            -1,
            height * width
        )

        attention = self.softmax(
            torch.bmm(query, key)
        )

        value = self.value(x).view(
            batch_size,
            channels,
            height * width
        )

        out = torch.bmm(
            value,
            attention.permute(0, 2, 1)
        )

        out = out.view(
            batch_size,
            channels,
            height,
            width
        )

        out = self.gamma * out + x

        return out
