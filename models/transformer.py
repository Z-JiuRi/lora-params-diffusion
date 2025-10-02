import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TransformerDiffusion(nn.Module):
    """Transformer扩散模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 输入输出维度
        if config.model.use_vae:
            self.in_dim = config.model.latent_dim
        else:
            from data.processor import LoRAProcessor
            processor = LoRAProcessor(config)
            self.in_dim = processor.total_dim
        
        self.out_dim = self.in_dim
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, config.model.transformer_dim))
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(config.model.transformer_dim, config.model.transformer_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model.transformer_dim * 4, config.model.transformer_dim)
        )
        
        # 输入投影
        self.input_proj = nn.Linear(self.in_dim, config.model.transformer_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.transformer_dim,
            nhead=config.model.transformer_heads,
            dim_feedforward=config.model.transformer_mlp_dim,
            dropout=config.model.transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.model.transformer_depth
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(config.model.transformer_dim, config.model.transformer_dim // 2),
            nn.SiLU(),
            nn.Linear(config.model.transformer_dim // 2, self.out_dim)
        )
    
    def forward(self, x, time, condition=None):
        batch_size = x.shape[0]
        
        # 输入投影
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, D)
        
        # 时间嵌入
        time_emb = self.time_embed(self.pos_embedding).expand(batch_size, -1, -1)
        x = x + time_emb
        
        # Transformer
        x = self.transformer(x)
        
        # 输出投影
        x = self.output_proj(x.squeeze(1))
        return x