import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """残差块 - 1D版本"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )
        
        self.residual_conv = nn.Linear(in_channels, out_channels) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        time_emb = self.time_mlp(time_emb)
        scale, shift = time_emb.chunk(2, dim=-1)
        h = h * (scale + 1) + shift
        
        h = self.block2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """注意力块"""
    
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # b, n, d = x.shape # 原始代码

        处理输入可能是2D或3D的情况
        if len(x.shape) == 3:
            b, n, d = x.shape
        else:  # 处理2D输入
            b, d = x.shape
            n = 1
            x = x.unsqueeze(1)  # 添加序列维度
            
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)

class UNet1D(nn.Module):
    """1D UNet for LoRA参数生成 - 完整版本"""
    
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
        
        # 时间嵌入
        time_dim = config.model.unet_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.model.unet_dim),
            nn.Linear(config.model.unet_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 初始投影
        self.init_proj = nn.Linear(self.in_dim, config.model.unet_dim)
        
        # 下采样路径
        self.downs = nn.ModuleList()
        channels = [config.model.unet_dim]
        
        for i, mult in enumerate(config.model.unet_dim_mults):
            dim_in = channels[-1]
            dim_out = config.model.unet_dim * mult
            
            layers = [
                ResidualBlock(dim_in, dim_out, time_dim, config.model.unet_dropout),
                ResidualBlock(dim_out, dim_out, time_dim, config.model.unet_dropout),
            ]
            
            # 在指定分辨率添加注意力
            if dim_out in config.model.unet_attention_resolutions:
                layers.append(AttentionBlock(dim_out, config.model.unet_heads, config.model.unet_dropout))
            
            # 下采样（除了最后一层）
            if i < len(config.model.unet_dim_mults) - 1:
                layers.append(nn.Linear(dim_out, dim_out // 2))
                channels.append(dim_out // 2)
            else:
                channels.append(dim_out)
            
            self.downs.append(nn.ModuleList(layers))
        
        # 中间块
        mid_dim = channels[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim, config.model.unet_dropout)
        self.mid_attn = AttentionBlock(mid_dim, config.model.unet_heads, config.model.unet_dropout)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim, config.model.unet_dropout)
        
        # 上采样路径
        self.ups = nn.ModuleList()
        
        for i, mult in enumerate(reversed(config.model.unet_dim_mults)):
            dim_in = channels.pop()
            dim_skip = channels[-1] if channels else 0
            dim_out = config.model.unet_dim * mult
            
            layers = []
            
            # 上采样投影
            if i > 0:
                layers.append(nn.Linear(dim_in + dim_skip, dim_in))
            
            layers.extend([
                ResidualBlock(dim_in, dim_out, time_dim, config.model.unet_dropout),
                ResidualBlock(dim_out, dim_out, time_dim, config.model.unet_dropout),
            ])
            
            # 注意力
            if dim_out in config.model.unet_attention_resolutions:
                layers.append(AttentionBlock(dim_out, config.model.unet_heads, config.model.unet_dropout))
            
            self.ups.append(nn.ModuleList(layers))
        
        # 最终投影
        self.final_proj = nn.Sequential(
            nn.GroupNorm(8, config.model.unet_dim),
            nn.SiLU(),
            nn.Linear(config.model.unet_dim, self.out_dim)
        )
    
    def forward(self, x, time, condition=None):
        # 时间嵌入
        t = self.time_mlp(time)
        
        # 初始投影
        x = self.init_proj(x)
        
        # 下采样
        skips = []
        for layers in self.downs:
            for layer in layers:
                if isinstance(layer, (ResidualBlock, AttentionBlock)):
                    x = layer(x, t) if hasattr(layer, 'time_mlp') else layer(x)
                else:
                    x = layer(x)
            skips.append(x)
        
        # 中间块
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # 上采样
        for layers in self.ups:
            # 跳跃连接
            if skips:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=-1)
            
            for layer in layers:
                if isinstance(layer, (ResidualBlock, AttentionBlock)):
                    x = layer(x, t) if hasattr(layer, 'time_mlp') else layer(x)
                else:
                    x = layer(x)
        
        # 最终投影
        return self.final_proj(x)