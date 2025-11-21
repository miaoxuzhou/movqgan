import torch
import torch.nn as nn
from contextlib import contextmanager

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) for model parameters.

    Maintains a shadow copy of the model parameters that are updated as:
        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warm_up: bool = True,
    ):
        super().__init__()

        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1.")
        
        self.warm_up = warm_up

        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if warm_up else torch.tensor(-1, dtype=torch.int))

        # mapping from model parameter names to shadow parameter names
        self.param_to_shadow_name = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # remove as '.'-character is not allowed in buffers
                sname = name.replace('.','')
                self.param_to_shadow_name.update({name:sname})
                self.register_buffer(sname,param.clone().detach().data)

        self.collected_params = []
    
    @torch.no_grad()
    def forward(self, model):
        decay = float(self.decay)

        if self.warm_up and int(self.num_updates) >= 0:
            self.num_updates += 1
            decay = min(float(self.decay), (1 + int(self.num_updates)) / (10 + int(self.num_updates)))
        
        one_minus_decay = 1.0 - decay

        params = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())

        for name, param in params.items():
            if param.requires_grad:
                # Handle DDP Wrapper
                name_stripped = name.replace('module.','')

                if name in self.param_to_shadow_name:
                    sname = self.param_to_shadow_name[name]
                elif name_stripped in self.param_to_shadow_name:
                    sname = self.param_to_shadow_name[name_stripped]
                else:
                    print(f"Warning: Parameter {name} not found in EMA model.")
                    continue
                
                if sname in shadow_params:
                    shadow_params[sname] = shadow_params[sname].to(
                        dtype=param.dtype, 
                        device=param.device
                    )
                    shadow_params[sname].sub_(
                        one_minus_decay * (shadow_params[sname] - param)
                    )
            else:
                name_stripped = name.replace('module.','')
                assert not name in self.param_to_shadow_name and not name_stripped in self.param_to_shadow_name, \
                f"Parameter {name} does not require gradients but is in EMA model."
    
    def copy_to(self, model):
        params = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())

        for name, param in params.items():
            if param.requires_grad:
                name_stripped = name.replace('module.','')
                
                if name in self.param_to_shadow_name:
                    sname = self.param_to_shadow_name[name]
                elif name_stripped in self.param_to_shadow_name:
                    sname = self.param_to_shadow_name[name_stripped]
                else:
                    continue
                
                if sname in shadow_params:
                    param.data.copy_(shadow_params[sname].data)
            else:
                name_stripped = name.replace('module.','')
                assert not name in self.param_to_shadow_name and not name_stripped in self.param_to_shadow_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        
    @contextmanager
    def apply_shadow(self, model):
        self.store(model.parameters())
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model.parameters())