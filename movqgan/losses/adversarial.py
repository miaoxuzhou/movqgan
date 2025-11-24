import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.relu(1. - logits_real)) + 
        torch.mean(F.relu(1. + logits_fake))
    )
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake))
    )
    return d_loss

class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GANs.
    """
    def __init__(self, loss_type='hinge'):
        """
        Initialize the adversarial loss module.

        Args:
            loss_type (str): Type of adversarial loss to use. Options are 'hinge' or 'vanilla'.
        """
        super(AdversarialLoss, self).__init__()

        if loss_type == 'hinge':
            self.d_loss_fn = hinge_d_loss
        elif loss_type == 'vanilla':
            self.d_loss_fn = vanilla_d_loss
        else:
            raise NotImplementedError(f"Adversarial loss type '{loss_type}' is not implemented.")

    def generator_loss(self, logits_fake):
        return -torch.mean(logits_fake)
    
    def discriminator_loss(self, logits_real, logits_fake):
        return self.d_loss_fn(logits_real, logits_fake)
    
    def forward(self, logits_real, logits_fake, mode='discriminator'):
        if mode == 'generator':
            return self.generator_loss(logits_fake)
        elif mode == 'discriminator':
            return self.discriminator_loss(logits_real, logits_fake)
        else:
            raise NotImplementedError(f"Adversarial loss mode '{mode}' is not implemented.")