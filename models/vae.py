import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class VAEEncoder(nn.Module):
    """VAE编码器 - 改进版本"""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        
        modules = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            modules.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # 潜在空间
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

class VAEDecoder(nn.Module):
    """VAE解码器 - 改进版本"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        
        modules = []
        in_dim = latent_dim
        
        for h_dim in reversed(hidden_dims):
            modules.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        modules.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*modules)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class LoRAVAE(nn.Module):
    """LoRA专用VAE - 完整版本"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 计算输入维度
        from data.processor import LoRAProcessor
        self.processor = LoRAProcessor(config)
        input_dim = self.processor.total_dim
        
        # 构建编码器和解码器
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.vae_hidden_dims,
            dropout=0.1
        )
        
        self.decoder = VAEDecoder(
            latent_dim=config.model.latent_dim,
            output_dim=input_dim,
            hidden_dims=config.model.vae_hidden_dims,
            dropout=0.1
        )
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, lora_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码LoRA矩阵到潜在空间"""
        flat = self.processor.flatten_lora(lora_dict)
        mu, log_var = self.encoder(flat)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从潜在空间解码LoRA矩阵"""
        flat = self.decoder(z)
        lora_dict = self.processor.unflatten_lora(flat)
        return lora_dict
    
    def forward(self, lora_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """前向传播"""
        z, mu, log_var = self.encode(lora_dict)
        recon_dict = self.decode(z)
        return recon_dict, mu, log_var
    
    def loss_function(self, recon_dict: Dict[str, torch.Tensor], 
                     original_dict: Dict[str, torch.Tensor],
                     mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """VAE损失函数"""
        # 重建损失
        recon_loss = 0
        for key in original_dict.keys():
            recon_loss += F.mse_loss(recon_dict[key], original_dict[key])
        recon_loss /= len(original_dict.keys())
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
        
        # 总损失
        total_loss = recon_loss + self.config.model.vae_beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }