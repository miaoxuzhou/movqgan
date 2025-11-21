import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
from argparse import ArgumentParser

from movqgan.data.dataset import LightningDataModule
from movqgan.util import instantiate_from_config
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Weights & Biases configuration
os.environ["WANDB_API_KEY"] = "7437e5599ac1fed693efa16749d40b2d88e00212"  
os.environ["WANDB_MODE"] = "online" # "online", "offline", or "disabled"

def run_train(config):
    """
    Run training with given configuration.
    
    Args:
        config: OmegaConf configuration dictionary
    """
    print("=" * 80)
    print("MoVQGAN Training")
    print("=" * 80)
    
    # Instantiate model from config
    print("\nLoading model...")
    model = instantiate_from_config(config['model'])
    print(f"Model loaded: {config['model']['target']}")
    
    # Instantiate data module
    print("\nLoading data...")
    data = LightningDataModule(config['data']['train'])
    print(f"Data loaded from: {config['data']['train']['df_path']}")
    print(f"  - Image size: {config['data']['train']['image_size']}")
    print(f"  - Batch size: {config['data']['train']['batch_size']}")
    
    # Create checkpoint directory if it doesn't exist
    print("\nSetting up checkpoints...")
    checkpoint_dir = config['ModelCheckpoint']['dirpath']
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Configure callbacks
    callbacks = [
        ModelCheckpoint(**config['ModelCheckpoint']),
        LearningRateMonitor(logging_interval='step')
    ]
    print(f"Callbacks configured: {len(callbacks)} callbacks")
    
    # Configure logger
    print("\nSetting up logging...")
    wandb_logger = WandbLogger(
        entity=config['wandb_entity_name'],
        project=config['wandb_project_name'],
        name=f"{config['wandb_project_name']}-{config['ModelCheckpoint']['dirpath'].split('/')[-1]}",
        save_dir=checkpoint_dir,
    )
    print(f"W&B logger: project='{config['wandb_project_name']}'")
    
    # Create PyTorch Lightning Trainer
    print("\nCreating trainer...")
    trainer = pl.Trainer(
        logger=wandb_logger, 
        callbacks=callbacks, 
        **config['trainer']
    )
    print(f"Trainer configured:")
    print(f"  - Accelerator: {config['trainer']['accelerator']}")
    print(f"  - Devices: {config['trainer']['devices']}")
    print(f"  - Strategy: {config['trainer']['strategy']}")
    print(f"  - Max steps: {config['trainer']['max_steps']}")
    
    # Determine checkpoint path for resuming
    if config['ckpt_path'] == '' or config['ckpt_path'] is None:
        ckpt_path = None
        print("\nTraining from scratch")
    else:
        ckpt_path = config['ckpt_path']
        print(f"\nResuming from checkpoint: {ckpt_path}")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.fit(model, data, ckpt_path=ckpt_path)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

def main():
    """Parse arguments and start training."""
    parser = ArgumentParser(description='Train MoVQGAN models')
    parser.add_argument(
        '--config', 
        type=str, required=True, 
        help='Path to YAML configuration file (e.g., configs/movqgan_67M.yaml)'
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    config = OmegaConf.load(args.config)
    
    # Start training
    run_train(config)

if __name__ == '__main__':
    main()