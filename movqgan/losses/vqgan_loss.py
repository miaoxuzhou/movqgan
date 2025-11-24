import torch
import torch.nn as nn

from .perceptual import ReconstructionLoss, PerceptualLoss
from .adversarial import AdversarialLoss
from movqgan.modules.discriminator.discriminator import NLayerDiscriminator, weights_init

class VQGANLoss(nn.Module):
    """
    Complete VQ-GAN loss combining:
    - Reconstruction loss (pixel + perceptual)
    - Adversarial loss (generator + discriminator)
    - Codebook loss (vector quantization)
    """
    
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=0.1,
        disc_weight=1.0,
        disc_factor=0.1,
        disc_loss="hinge",
        disc_num_layers=3,
        disc_in_channels=3,
        disc_ndf=64,
        disc_conditional=False,
        use_actnorm=False
    ):
        """
        Args:
            disc_start: Global step to start discriminator training
            codebook_weight: Weight for codebook loss
            pixelloss_weight: Weight for pixel reconstruction loss
            perceptual_weight: Weight for perceptual loss
            disc_weight: Weight for discriminator loss
            disc_factor: Additional scaling factor for discriminator
            disc_loss: Type of adversarial loss ("hinge" or "vanilla")
            disc_num_layers: Number of layers in discriminator
            disc_in_channels: Input channels for discriminator
            disc_ndf: Number of discriminator filters
            disc_conditional: Whether to use conditional discriminator
            use_actnorm: Whether to use ActNorm in discriminator
        """
        super().__init__()
        
        # Codebook loss
        self.codebook_weight = codebook_weight

        # Loss weights
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        # Discriminator settings
        self.discriminator_iter_start = disc_start
        self.disc_conditional = disc_conditional
        
        # Reconstruction loss
        self.reconstruction_loss = ReconstructionLoss()
        # Perceptual loss
        self.perceptual_loss = PerceptualLoss()
        # Adversarial loss
        self.adversarial_loss = AdversarialLoss(loss_type=disc_loss)
        
        # Discriminator network
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        ).apply(weights_init)
        
        print(f"VQGANLoss initialized with {disc_loss} adversarial loss.")
    
    def calculate_adaptive_weight(
        self, 
        nll_loss, 
        g_loss, 
        last_layer
    ):
        """
        Calculate adaptive weight for balancing reconstruction and adversarial loss.
        
        Args:
            nll_loss: Reconstruction loss
            g_loss: Generator adversarial loss
            last_layer: Last layer of decoder for gradient calculation
            
        Returns:
            Adaptive weight for discriminator loss
        """

        # torch.autograd.grad(loss, params) 计算 loss 对 params 的梯度
        # retain_graph=True 保留计算图，以便后续计算梯度
        # 如果外部传入了 last_layer，则使用它，否则使用 self.last_layer[0]
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        # Calculate L2 norms of both gradients and compute their ratio
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        # Clamp the adaptive weight to avoid extreme values
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # Scale by predefined discriminator weight
        d_weight = d_weight * self.disc_weight

        return d_weight
    
    def compute_generator_loss(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        global_step,
        last_layer=None,
        cond=None,
        split="train"
    ):
        """
        Compute loss for generator (autoencoder) update.
        
        Args:
            codebook_loss: Vector quantization loss
            inputs: Original images
            reconstructions: Reconstructed images
            global_step: Current training step
            last_layer: Last decoder layer for adaptive weighting
            cond: Conditional input for discriminator (if conditional)
            split: "train" or "val"
            
        Returns:
            total_loss: Combined generator loss
            log_dict: Dictionary of loss components for logging
        """
        # 1. Reconstruction loss (pixel + perceptual)
        rec_loss = self.reconstruction_loss(inputs, reconstructions)
        p_loss = self.perceptual_loss(inputs, reconstructions)

        total_loss = (
            self.pixelloss_weight * rec_loss + 
            self.perceptual_weight * p_loss
        )

        # 2. Generator adversarial loss
        # Get discriminator output for reconstructions
        if cond is None:
            assert not self.disc_conditional
            logits_fake = self.discriminator(
                reconstructions.contiguous()
            )
        else:
            assert self.disc_conditional
            logits_fake = self.discriminator(
                torch.cat((reconstructions.contiguous(), cond), dim=1)
            )
        
        g_loss = self.adversarial_loss.generator_loss(logits_fake)
        
        # 3. Apply adaptive weighting for adversarial loss
        # Only apply after discriminator has started training
        if global_step >= self.discriminator_iter_start and last_layer is not None:
            try:
                d_weight = self.calculate_adaptive_weight(total_loss, g_loss, last_layer)
            except RuntimeError:
                # If adaptive weight calculation fails, use fixed weight
                d_weight = torch.tensor(self.disc_weight, device=total_loss.device)
        else:
            d_weight = torch.tensor(0.0, device=total_loss.device)  # Don't use discriminator before disc_start
        
        # 4. Combine all losses
        total_loss += (
            d_weight * self.disc_factor * g_loss +
            self.codebook_weight * codebook_loss.mean()
        )
        
        # 5. Prepare logging dictionary
        log_dict = {
            f"{split}/total_loss": total_loss.clone().detach(),
            f"{split}/quant_loss": codebook_loss.detach().mean(),
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/p_loss": p_loss.detach(),
            f"{split}/g_loss": g_loss.detach(),
            f"{split}/d_weight": d_weight.detach(),
        }
        
        return total_loss, log_dict
    
    def compute_discriminator_loss(
        self,
        inputs,
        reconstructions,
        cond=None,
        split="train"
    ):
        """
        Compute loss for discriminator update.
        
        Args:
            inputs: Original images
            reconstructions: Reconstructed images
            cond: Conditional input for discriminator (if conditional)
            split: "train" or "val"
            
        Returns:
            disc_loss: Discriminator loss
            log_dict: Dictionary of loss components for logging
        """
        # Get discriminator outputs
        if cond is None:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        else:
            logits_real = self.discriminator(
                torch.cat((inputs.contiguous().detach(), cond), dim=1)
            )
            logits_fake = self.discriminator(
                torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
            )
        
        # Compute discriminator loss
        disc_loss = self.adversarial_loss.discriminator_loss(logits_real, logits_fake)
        
        # Prepare logging dictionary
        log_dict = {
            f"{split}/disc_loss": disc_loss.clone().detach(),
            f"{split}/logits_real": logits_real.detach().mean(),
            f"{split}/logits_fake": logits_fake.detach().mean(),
        }
        
        return disc_loss, log_dict
    
    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train"
    ):
        """
        Forward pass - routes to generator or discriminator loss based on optimizer_idx.
        
        Args:
            codebook_loss: Vector quantization loss
            inputs: Original images
            reconstructions: Reconstructed images
            optimizer_idx: 0 for generator, 1 for discriminator
            global_step: Current training step
            last_layer: Last decoder layer for adaptive weighting
            cond: Conditional input for discriminator (if conditional)
            split: "train" or "val"
            
        Returns:
            loss: Computed loss value
            log_dict: Dictionary of loss components for logging
        """
        if optimizer_idx == 0:
            # Generator (autoencoder) update
            return self.compute_generator_loss(
                codebook_loss=codebook_loss,
                inputs=inputs,
                reconstructions=reconstructions,
                global_step=global_step,
                last_layer=last_layer,
                cond=cond,
                split=split
            )
        elif optimizer_idx == 1:
            # Discriminator update
            return self.compute_discriminator_loss(
                inputs=inputs,
                reconstructions=reconstructions,
                cond=cond,
                split=split
            )
        else:
            raise ValueError(f"Invalid optimizer_idx: {optimizer_idx}")