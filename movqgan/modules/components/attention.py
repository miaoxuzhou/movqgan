"""
Attention mechanisms for MoVQGAN.

This module contains various attention implementations used in the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from .normalization import Normalize, SpatialNorm

class AttnBlock(nn.Module):
    """
    Standard self-attention block.
    
    Uses 1x1 convolutions for Q, K, V projections and applies
    scaled dot-product attention.
    
    Args:
        in_channels: Number of input channels
        
    Example:
        >>> attn = AttnBlock(in_channels=256)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 256, 32, 32])
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # Use GroupNorm for normalization
        self.norm = Normalize(in_channels=in_channels)
        
        # 1x1 convolutions for Q, K, V projections
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        # Output projection
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        residual = x
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Compute Q, K, V
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)
        
        # Compute attention
        b, c, h, w = q.shape
        
        # Reshape for attention computation: (B, C, H*W) -> (B, H*W, C)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
        k = k.reshape(b, c, h * w)  # (B, C, H*W)
        
        # Compute attention weights: (B, H*W, H*W)
        attn_weights = torch.bmm(q, k)  # (B, H*W, H*W)
        attn_weights = attn_weights * (int(c) ** (-0.5))  # Scale
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # Apply attention to values
        v = v.reshape(b, c, h * w)  # (B, C, H*W)
        attn_weights = attn_weights.permute(0, 2, 1)  # (B, H*W, H*W)
        out = torch.bmm(v, attn_weights)  # (B, C, H*W)
        out = out.reshape(b, c, h, w)  # (B, C, H, W)
        
        # Output projection
        out = self.proj_out(out)
        
        # Add residual
        return residual + out

class SpatialAttnBlock(nn.Module):
    """
    Spatial attention block with modulation support.
    
    This is an attention block that can be modulated by additional
    conditioning information (e.g., quantized codes).
    
    Args:
        in_channels: Number of input channels
        zq_channels: Number of channels in conditioning input (optional)
        add_conv: Whether to add extra convolution for conditioning
        
    Example:
        >>> attn = SpatialAttnBlock(in_channels=256, zq_channels=4)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> zq = torch.randn(2, 4, 8, 8)
        >>> out = attn(x, zq)
    """
    
    def __init__(
        self, 
        in_channels: int,
        zq_channels: Optional[int] = None,
        add_conv: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.zq_channels = zq_channels
        self.add_conv = add_conv
        
        # Normalization with spatial modulation
        self.norm = SpatialNorm(
            f_channels=in_channels,
            zq_channels=zq_channels,
            add_conv=add_conv
        )
        
        # Q, K, V projections
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        zq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply spatial attention with optional modulation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            zq: Optional conditioning tensor for modulation
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        residual = x
        
        # Apply normalization (with modulation)
        x_norm = self.norm(x, zq)
        
        # Compute Q, K, V
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)
        
        # Compute attention
        b, c, h, w = q.shape
        
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        
        attn_weights = torch.bmm(q, k)
        attn_weights = attn_weights * (int(c) ** (-0.5))
        attn_weights = F.softmax(attn_weights, dim=2)
        
        v = v.reshape(b, c, h * w)
        attn_weights = attn_weights.permute(0, 2, 1)
        out = torch.bmm(v, attn_weights)
        out = out.reshape(b, c, h, w)
        
        out = self.proj_out(out)
        
        return residual + out