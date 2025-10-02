import torch
import torch.nn as nn
from typing import Iterable, Optional

class EMAModel:
    """
    Exponential Moving Average for models.
    Maintains moving averages of model parameters using exponential decay.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create shadow parameters
        self.shadow_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
        
        # Create shadow buffers (for BatchNorm, etc.)
        self.shadow_buffers = {}
        for name, buffer in self.model.named_buffers():
            self.shadow_buffers[name] = buffer.data.clone()
    
    def update(self):
        """Update the shadow parameters using exponential moving average."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = self.shadow_params[name].to(self.device)
                    self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply shadow parameters to the model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow_params[name])
            
            for name, buffer in self.model.named_buffers():
                buffer.data.copy_(self.shadow_buffers[name])
    
    def restore(self):
        """Restore original parameters to the model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.original_params[name])
            
            for name, buffer in self.model.named_buffers():
                buffer.data.copy_(self.original_buffers[name])
    
    def store_original(self):
        """Store original parameters for later restoration."""
        self.original_params = {}
        self.original_buffers = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_params[name] = param.data.clone()
        
        for name, buffer in self.model.named_buffers():
            self.original_buffers[name] = buffer.data.clone()
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params,
            'shadow_buffers': self.shadow_buffers,
            'original_params': getattr(self, 'original_params', {}),
            'original_buffers': getattr(self, 'original_buffers', {})
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']
        self.shadow_buffers = state_dict['shadow_buffers']
        
        if 'original_params' in state_dict:
            self.original_params = state_dict['original_params']
        if 'original_buffers' in state_dict:
            self.original_buffers = state_dict['original_buffers']
    
    def to(self, device):
        """Move EMA parameters to device."""
        self.device = device
        for name in self.shadow_params:
            self.shadow_params[name] = self.shadow_params[name].to(device)
        for name in self.shadow_buffers:
            self.shadow_buffers[name] = self.shadow_buffers[name].to(device)