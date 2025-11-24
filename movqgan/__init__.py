import os
import torch

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from .models.base_vq import BaseVQModel
from .models.vq import VQ
from .models.movq import MOVQ
from .models.gumbel_vq import GumbelVQ, GumbelQuantize
from .models.ema import EMA
from .util import get_ckpt_path, get_obj_from_str, instantiate_from_config

__version__ = '0.1.0'

__all__ = [
    # Models
    'BaseVQModel', 
    'VQ', 
    'MOVQ', 
    'GumbelVQ',
    'GumbelQuantize',

    # Utilities
    'EMA',
    'get_ckpt_path',
    'get_obj_from_str',
    'instantiate_from_config',

    # Pretrained model loading
    'get_movqgan_model',
    
    # Metadata
    '__version__'
]

# Pretrained model configurations
MODELS = {
    '67M': dict(
        description='MOVQGAN with 67M parameters',
        model_params=dict(
            ddconfig={
                "double_z": False,          # Don't output mean and variation (not VAE)
                "z_channels": 4,            # Number of channels in latent space
                "resolution": 256,          # Input resolution
                "in_channels": 3,           # Input image channels (RGB)
                "out_ch": 3,                # Output image channels (RGB)
                "ch": 128,                  # Base channel count
                "ch_mult": [1, 2, 2, 4],    # Channel multipliers at each resolution
                "num_res_blocks": 2,        # Number of residual blocks at each resolution
                "attn_resolutions": [32],   # Resolutions at which attention is applied
                "dropout": 0.0,             # Dropout rate
            },
            n_embed=16384,               # Codebook size
            embed_dim=4,                 # Embedding dimension
        ),
        repo_id='ai-forever/MoVQGAN',
        filename='movqgan_67M.ckpt',
        authors='SberAI',
        full_description='',
    ),
    '102M': dict(
        description='MOVQGAN with 102M parameters',
        model_params=dict(
            ddconfig={
                "double_z": False,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 2, 4],
                "num_res_blocks": 4,          # Number of residual blocks at each resolution
                "attn_resolutions": [32],
                "dropout": 0.0,
            },
            n_embed=16384,
            embed_dim=4,
        ),
        repo_id='ai-forever/MoVQGAN',
        filename='movqgan_102M.ckpt',
        authors='SberAI',
        full_description='',
    ),
    '270M': dict(
        description='',
        model_params=dict(
            ddconfig={
                "double_z": False,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 256,                     # Base channel count
                "ch_mult": [1, 2, 2, 4],
                "num_res_blocks": 2,           # Number of residual blocks at each resolution
                "attn_resolutions": [32],
                "dropout": 0.0,
            },
            n_embed=16384,
            embed_dim=4,
        ),
        repo_id='ai-forever/MoVQGAN',
        filename='movqgan_270M.ckpt',
        authors='SberAI',
        full_description='',
    )
}

# Model loading functions
def get_movqgan_model(
    name: str, 
    pretrained: bool = True, 
    device: str = 'cuda', 
    cache_dir: str = '/tmp/movqgan', 
    **model_kwargs
) -> MOVQ:
    
    # Validate model name
    assert name in MODELS
    
    # Check HuggingFace Hub availability
    if pretrained and not HF_HUB_AVAILABLE:
        raise ImportError("Please install the 'huggingface_hub' package to load pretrained models.")
    
    # Get model configuration
    config = MODELS[name].copy()
    config['model_params'].update(model_kwargs)

    # Add default parameters if not provided
    # We don't need these for inference, but they are required by the constructor
    model_params = config['model_params'].copy()
    if 'learning_rate' not in model_params:
        model_params['learning_rate'] = 1e-4  # Default learning rate
    if 'lossconfig' not in model_params:
        model_params['lossconfig'] = dict(
            target='movqgan.losses.vqgan_loss.VQGANLoss',
            params=dict(
                disc_start=0,
                disc_weight=1.0,
                disc_loss='hinge',
            )
        )
    
    # Instantiate model
    model = MOVQ(**model_params)

    # Load pretrained weights if requested
    if pretrained:
        cache_dir = os.path.join(cache_dir, name)
        os.makedirs(cache_dir, exist_ok=True)

        checkpoint_path = hf_hub_download(
            repo_id=config['repo_id'], 
            filename=config['filename'],
            cache_dir=cache_dir, 
        )

        # Load checkpoint
        print(f"Loading pretrained weights from {checkpoint_path}...")
        
        checkpoint = torch.load(
            checkpoint_path, 
            map_location='cpu'
        )

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading pretrained weights: {unexpected}")

    model.eval()
    model = model.to(device)
    return model