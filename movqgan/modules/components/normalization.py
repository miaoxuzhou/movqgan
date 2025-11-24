"""
Normalization layers for MoVQGAN.

This module contains various normalization techniques including
spatial normalization with modulation for the MoVQ decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Convenience function for standard GroupNorm
def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    """
    Create standard GroupNorm layer.
    
    This is a convenience function matching the original codebase naming.
    
    Args:
        in_channels: Number of input channels
        num_groups: Number of groups
        
    Returns:
        GroupNorm layer
    """
    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True
    )

class SpatialNorm(nn.Module):
    """
    Spatial normalization with modulation.
    
    This normalization layer applies GroupNorm and then modulates
    the normalized features using spatially-varying scale and bias
    derived from conditioning information (e.g., quantized codes).
    
    This is a key component of MoVQGAN that allows the decoder to be
    conditioned on the quantized representation.
    
    Args:
        f_channels: Number of feature channels to normalize
        zq_channels: Number of channels in conditioning input
        norm_layer: Base normalization layer (default: GroupNorm)
        freeze_norm_layer: Whether to freeze the base norm layer
        add_conv: Whether to add extra convolution for conditioning
        num_groups: Number of groups for GroupNorm
        eps: Epsilon for numerical stability
        affine: Whether to use learnable affine parameters
        
    Example:
        >>> norm = SpatialNorm(f_channels=128, zq_channels=4)
        >>> features = torch.randn(2, 128, 32, 32)
        >>> condition = torch.randn(2, 4, 8, 8)
        >>> normalized = norm(features, condition)
    """
    
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        norm_layer: type = nn.GroupNorm,
        freeze_norm_layer: bool = False,
        add_conv: bool = False,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()
        
        # Base normalization layer
        self.norm_layer = norm_layer(
            num_channels=f_channels,
            num_groups=num_groups,
            eps=eps,
            affine=affine
        )
        
        # Freeze base norm if requested
        if freeze_norm_layer:
            for param in self.norm_layer.parameters():
                param.requires_grad = False
        
        # Optional additional convolution for conditioning
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv2d(
                zq_channels, 
                zq_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1
            )
        
        # Modulation parameters (scale and bias)
        self.conv_y = nn.Conv2d(
            zq_channels, 
            f_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.conv_b = nn.Conv2d(
            zq_channels, 
            f_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
    
    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial normalization with modulation.
        
        Args:
            f: Feature tensor to normalize, shape (B, C_f, H_f, W_f)
            zq: Conditioning tensor, shape (B, C_zq, H_zq, W_zq)
            
        Returns:
            Normalized and modulated features, shape (B, C_f, H_f, W_f)
        """
        # Get feature spatial size
        f_size = f.shape[-2:]
        
        # Resize conditioning to match feature size
        zq = F.interpolate(zq, size=f_size, mode='nearest')
        
        # Apply optional convolution to conditioning
        if self.add_conv:
            zq = self.conv(zq)
        
        # Normalize features
        norm_f = self.norm_layer(f)
        
        # Compute scale and bias from conditioning
        scale = self.conv_y(zq)
        bias = self.conv_b(zq)
        
        # Apply modulation: normalized * scale + bias
        return norm_f * scale + bias

class ActNorm(nn.Module):
    """
    Activation Normalization.
    
    Normalizes activations using data-dependent initialization.
    After initialization, acts as an affine transformation.
    
    Args:
        num_features: Number of features to normalize
        logdet: Whether to compute log determinant (for normalizing flows)
        affine: Whether to use affine transformation
        allow_reverse_init: Whether to allow reverse initialization
        
    Example:
        >>> norm = ActNorm(256)
        >>> x = torch.randn(2, 256, 32, 32)
        >>> out = norm(x)
    """
    
    def __init__(
        self,
        num_features: int,
        logdet: bool = False,
        affine: bool = True,
        allow_reverse_init: bool = False,
    ):
        super().__init__()
        assert affine, "ActNorm requires affine=True"
        
        self.logdet = logdet
        self.allow_reverse_init = allow_reverse_init
        
        # Learnable parameters
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        
        # Initialization flag
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
    
    def initialize(self, input: torch.Tensor) -> None:
        """
        Initialize parameters based on input statistics.
        
        Args:
            input: Input tensor for initialization
        """
        with torch.no_grad():
            # Flatten spatial dimensions
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            
            # Compute mean and std
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            
            # Set parameters
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
    
    def forward(self, input: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Apply activation normalization.
        
        Args:
            input: Input tensor
            reverse: Whether to apply reverse transformation
            
        Returns:
            Normalized tensor (and optionally log determinant)
        """
        if reverse:
            return self.reverse(input)
        
        # Handle 2D inputs
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        
        _, _, height, width = input.shape
        
        # Initialize on first forward pass
        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        
        # Apply transformation
        h = self.scale * (input + self.loc)
        
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet
        
        return h
    
    def reverse(self, output: torch.Tensor) -> torch.Tensor:
        """
        Apply reverse transformation.
        
        Args:
            output: Output tensor to reverse
            
        Returns:
            Reversed tensor
        """
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)
        
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        
        h = output / self.scale - self.loc
        
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        
        return h

def get_normalization_layer(
    norm_type: str,
    num_channels: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create normalization layers.
    
    Args:
        norm_type: Type of normalization ('group', 'batch', 'layer', 'act', 'spatial')
        num_channels: Number of channels
        **kwargs: Additional arguments for specific norm types
        
    Returns:
        Normalization layer
        
    Example:
        >>> norm = get_normalization_layer('group', 128, num_groups=32)
        >>> norm = get_normalization_layer('spatial', 128, zq_channels=4)
        >>> norm = get_normalization_layer('layer', 256)
    """
    if norm_type == 'group':
        num_groups = kwargs.get('num_groups', 32)
        return nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=True)
    
    elif norm_type == 'batch':
        return nn.BatchNorm2d(num_channels)
    
    elif norm_type == 'act':
        return ActNorm(num_channels, **kwargs)
    
    elif norm_type == 'spatial':
        zq_channels = kwargs.get('zq_channels')
        if zq_channels is None:
            raise ValueError("zq_channels must be provided for spatial normalization")
        return SpatialNorm(num_channels, zq_channels, **kwargs)
    
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")
    