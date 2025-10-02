import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from configs.config_base import Config
from configs.model_configs import get_medium_config
from models.vae import LoRAVAE
from data.dataloader import create_train_val_dataloaders
from utils.utils import setup_logging, set_seed, get_device, model_summary, create_experiment_dir

logger = logging.getLogger(__name__)

class VAETrainer:
    """VAE训练器"""
    
    def __init__(self, config, train_loader, val_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device(config)
        
        # 创建VAE模型
        self.vae = LoRAVAE(config).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.train.num_epochs
        )
        
        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.info("VAE Trainer initialized")
        model_summary(self.vae)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch in self.train_loader:
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            recon_dict, mu, log_var = self.vae(batch)
            
            # 计算损失
            losses = self.vae.loss_function(recon_dict, batch, mu, log_var)
            loss = losses['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            total_recon_loss += losses['recon_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
        
        # 更新学习率
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        if self.val_loader is None:
            return None, None, None
            
        self.vae.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            recon_dict, mu, log_var = self.vae(batch)
            losses = self.vae.loss_function(recon_dict, batch, mu, log_var)
            
            total_loss += losses['total_loss'].item()
            total_recon_loss += losses['recon_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_recon_loss = total_recon_loss / len(self.val_loader)
        avg_kl_loss = total_kl_loss / len(self.val_loader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def save_checkpoint(self, path, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'vae_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'vae_best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
    
    def train(self):
        """训练主循环"""
        logger.info("Starting VAE training...")
        
        for epoch in range(self.epoch, self.config.train.num_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss, train_recon, train_kl = self.train_epoch()
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f} "
                       f"(Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
            
            # 验证
            if self.val_loader is not None:
                val_loss, val_recon, val_kl = self.validate()
                logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f} "
                           f"(Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
                
                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(f'vae_checkpoint_epoch_{epoch}.pth', is_best=True)
            
            # 保存检查点
            if epoch % 10 == 0:
                self.save_checkpoint(f'vae_checkpoint_epoch_{epoch}.pth')
        
        logger.info("VAE training completed!")

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--exp_name", type=str, default="vae", help="Experiment name")
    args = parser.parse_args()
    
    # 配置
    config = get_medium_config()
    config.train.exp_name = args.exp_name
    config.train.num_epochs = 200  # VAE通常需要较少的训练轮数
    
    # 设置
    set_seed(config.system.seed)
    create_experiment_dir("./exps", config.train.exp_name)
    setup_logging(f"./exps/{config.train.exp_name}/logs")

    # 创建数据加载器
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_train_val_dataloaders(config)
    
    # 创建训练器
    logger.info("Creating VAE trainer...")
    trainer = VAETrainer(config, train_loader, val_loader)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()