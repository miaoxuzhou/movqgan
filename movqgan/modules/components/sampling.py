"""
Sampling layers for MoVQGAN.

This module contains upsampling and downsampling operations
used in the encoder and decoder architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

class Upsample(nn.Module):
    """
    Upsampling layer with optional convolution.
    
    Upsamples spatial dimensions by a factor of 2 using nearest neighbor
    interpolation, optionally followed by a convolution.
    
    Args:
        in_channels: Number of input channels
        with_conv: Whether to apply convolution after upsampling
        
    Example:
        >>> upsample = Upsample(in_channels=128, with_conv=True)
        >>> x = torch.randn(2, 128, 16, 16)
        >>> out = upsample(x)
        >>> out.shape
        torch.Size([2, 128, 32, 32])
    """
    
    def __init__(self, in_channels: int, with_conv: bool = False):
        super().__init__()
        self.with_conv = with_conv
        
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Upsampled tensor of shape (B, C, 2*H, 2*W)
        """
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        
        if self.with_conv:
            x = self.conv(x)
        
        return x

class Downsample(nn.Module):
    """
    Downsampling layer with optional convolution.
    
    Downsamples spatial dimensions by a factor of 2, either using
    a strided convolution or average pooling.
    
    Args:
        in_channels: Number of input channels
        with_conv: Whether to use strided conv (True) or avg pooling (False)
        
    Example:
        >>> downsample = Downsample(in_channels=128, with_conv=True)
        >>> x = torch.randn(2, 128, 32, 32)
        >>> out = downsample(x)
        >>> out.shape
        torch.Size([2, 128, 16, 16])
    """
    
    def __init__(self, in_channels: int, with_conv: bool = False):
        super().__init__()
        self.with_conv = with_conv
        
        if self.with_conv:
            # Use strided convolution for downsampling
            # No asymmetric padding in PyTorch, so we pad manually
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Downsampled tensor of shape (B, C, H//2, W//2)
        """
        if self.with_conv:
            # Manual padding: (left, right, top, bottom)
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        return x