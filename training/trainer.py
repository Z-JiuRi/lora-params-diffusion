import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional

from models.ddpm import DDPM
from models.unet import UNet1D
from models.transformer import TransformerDiffusion
from models.vae import LoRAVAE
from data.processor import LoRAProcessor
from training.ema import EMAModel

logger = logging.getLogger(__name__)

class DiffusionTrainer:
    """扩散模型训练器"""
    
    def __init__(self, config, train_loader, val_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = self._setup_device()
        self.processor = LoRAProcessor(config)
        
        # 创建模型
        self.model = self._create_model()
        self.ddpm = DDPM(config.model.timesteps, self.model).to(self.device)
        
        # VAE（如果使用）
        self.vae = None
        if config.model.use_vae:
            self.vae = LoRAVAE(config).to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        self.scaler = GradScaler() if config.system.mixed_precision else None
        
        # EMA
        self.ema_model = None
        if config.train.use_ema:
            self.ema_model = EMAModel(
                self.ddpm.model,
                decay=config.train.ema_decay,
                device=self.device
            )
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def _setup_device(self):
        if self.config.system.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.system.device)
    
    def _create_model(self):
        if self.config.model.model_type == "unet":
            return UNet1D(self.config)
        elif self.config.model.model_type == "transformer":
            return TransformerDiffusion(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model.model_type}")
    
    def _create_optimizer(self):
        params = list(self.ddpm.parameters())
        if self.vae:
            params += list(self.vae.parameters())
            
        if self.config.train.optimizer == "adam":
            return optim.Adam(params, lr=self.config.train.learning_rate)
        elif self.config.train.optimizer == "adamw":
            return optim.AdamW(
                params, 
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay
            )
        else:
            return optim.SGD(params, lr=self.config.train.learning_rate)
    
    def train_epoch(self, epoch):
        self.ddpm.train()
        if self.vae:
            self.vae.train()
            
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 准备输入数据
            if self.vae:
                with torch.no_grad():
                    z, _, _ = self.vae.encode(batch)
                x_start = z
            else:
                x_start = self.processor.flatten_lora(batch)
            
            # 随机时间步
            batch_size = x_start.shape[0]
            t = torch.randint(0, self.config.model.timesteps, (batch_size,), device=self.device)
            
            # 训练步骤
            loss = self.train_step(x_start, t)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            self.global_step += 1
        
        return total_loss / len(self.train_loader)
    
    def train_step(self, x_start, t):
        self.optimizer.zero_grad()
        
        if self.scaler:
            with autocast():
                loss = self.ddpm.p_losses(x_start, t, loss_type=self.config.model.loss_type)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.ddpm.p_losses(x_start, t, loss_type=self.config.model.loss_type)
            loss.backward()
            
            if self.config.train.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddpm.parameters(), 
                    self.config.train.gradient_clip
                )
            
            self.optimizer.step()
        
        # 更新EMA
        if self.ema_model:
            self.ema_model.update()
            
        return loss
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
            
        self.ddpm.eval()
        if self.vae:
            self.vae.eval()
            
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.vae:
                z, _, _ = self.vae.encode(batch)
                x_start = z
            else:
                x_start = self.processor.flatten_lora(batch)
                
            batch_size = x_start.shape[0]
            t = torch.randint(0, self.config.model.timesteps, (batch_size,), device=self.device)
            
            loss = self.ddpm.p_losses(x_start, t, loss_type=self.config.model.loss_type)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, path):
        checkpoint = {
            'global_step': self.global_step,
            'ddpm_state': self.ddpm.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.vae:
            checkpoint['vae_state'] = self.vae.state_dict()
        if self.ema_model:
            checkpoint['ema_state'] = self.ema_model.state_dict()
        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.ddpm.load_state_dict(checkpoint['ddpm_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.vae and 'vae_state' in checkpoint:
            self.vae.load_state_dict(checkpoint['vae_state'])
        if self.ema_model and 'ema_state' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state'])
        if self.scaler and 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
            
        logger.info(f"Loaded checkpoint from {path}")
    
    def train(self):
        logger.info("Starting training...")
        
        for epoch in range(self.config.train.num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate()
            if val_loss is not None:
                logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # 早停
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pt")
                else:
                    self.patience_counter += 1
                    
                if self.config.train.early_stopping and self.patience_counter >= self.config.train.patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed!")