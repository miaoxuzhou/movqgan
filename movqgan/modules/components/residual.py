"""
Residual blocks for MoVQGAN.

This module contains various residual block implementations used
throughout the encoder and decoder architectures.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from .normalization import Normalize, SpatialNorm


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return x * torch.sigmoid(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings.
    
    Args:
        timesteps: 1D tensor of timesteps
        embedding_dim: Dimension of the embedding
    
    Returns:
        Embedding tensor of shape (len(timesteps), embedding_dim)
    """

    # Make sure timesteps is 1D
    assert len(timesteps.shape) == 1

    # Compute frequencies
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)

    # Generate all frequencies of shape (half_dim,)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    # Compute angles of shape (len(timesteps), half_dim)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]

    # Apply sine and cosine functions, resulting in shape (len(timesteps), dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    
    return emb

class ResnetBlock(nn.Module):
    """
    Standard residual block with optional time embedding.
    
    This is the basic building block used in the encoder.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (defaults to in_channels)
        conv_shortcut: Whether to use convolutional shortcut
        dropout: Dropout probability
        temb_channels: Number of time embedding channels (0 to disable)
        
    Example:
        >>> block = ResnetBlock(in_channels=128, out_channels=256, dropout=0.1)
        >>> x = torch.randn(2, 128, 32, 32)
        >>> out = block(x, temb=None)
        >>> out.shape
        torch.Size([2, 256, 32, 32])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        
        # First conv block
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Time embedding projection (if enabled)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.temb_proj = None
        
        # Second conv block
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
    
    def forward(
        self, 
        x: torch.Tensor, 
        temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            temb: Optional time embedding of shape (B, C_temb)
            
        Returns:
            Output tensor of shape (B, C_out, H, W)
        """
        residual = x
        
        # First conv block
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        # Add time embedding if provided
        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        
        # Second conv block
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Apply shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        
        return residual + h

class SpatialResnetBlock(nn.Module):
    """
    Residual block with spatial normalization (modulation).
    
    This block is used in the MoVQ decoder where features are
    modulated by the quantized codes.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (defaults to in_channels)
        conv_shortcut: Whether to use convolutional shortcut
        dropout: Dropout probability
        temb_channels: Number of time embedding channels (0 to disable)
        zq_ch: Number of channels in quantized code (for modulation)
        add_conv: Whether to add extra conv in SpatialNorm
        
    Example:
        >>> block = SpatialResnetBlock(
        ...     in_channels=128, 
        ...     out_channels=256, 
        ...     zq_ch=4
        ... )
        >>> x = torch.randn(2, 128, 32, 32)
        >>> zq = torch.randn(2, 4, 8, 8)
        >>> out = block(x, temb=None, zq=zq)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        
        # First conv block with spatial normalization
        self.norm1 = SpatialNorm(
            f_channels=in_channels,
            zq_channels=zq_ch,
            add_conv=add_conv
        )
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Time embedding projection (if enabled)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.temb_proj = None
        
        # Second conv block with spatial normalization
        self.norm2 = SpatialNorm(
            f_channels=self.out_channels,
            zq_channels=zq_ch,
            add_conv=add_conv
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
    
    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        zq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through spatial residual block.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            temb: Optional time embedding of shape (B, C_temb)
            zq: Quantized code for modulation of shape (B, C_zq, H_zq, W_zq)
            
        Returns:
            Output tensor of shape (B, C_out, H, W)
        """
        residual = x
        
        # First conv block with spatial modulation
        h = self.norm1(x, zq)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        # Add time embedding if provided
        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        
        # Second conv block with spatial modulation
        h = self.norm2(h, zq)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Apply shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        
        return residual + h