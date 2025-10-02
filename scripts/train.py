import os
import logging
from pathlib import Path

from configs.config_base import Config
from configs.model_configs import get_small_config, get_medium_config
from data.dataloader import create_train_val_dataloaders
from training.trainer import DiffusionTrainer
from utils.utils import setup_logging, set_seed, create_experiment_dir

def main():
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    args = parser.parse_args()
    
    # 设置
    config = get_small_config()
    config.train.exp_name = args.exp_name
    set_seed(config.system.seed)
    create_experiment_dir("./exps", config.train.exp_name)
    setup_logging(f"./exps/{config.train.exp_name}/logs")
    
    logger = logging.getLogger(__name__)
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_train_val_dataloaders(config)
    
    # 创建训练器
    logger.info("Creating trainer...")
    trainer = DiffusionTrainer(config, train_loader, val_loader)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()