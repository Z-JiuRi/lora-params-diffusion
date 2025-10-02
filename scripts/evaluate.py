import torch
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from configs.config_base import Config
from models.ddpm import DDPM
from models.unet import UNet1D
from models.vae import LoRAVAE
from data.processor import LoRAProcessor
from utils.utils import set_seed, get_device, compute_frechet_distance
from data.dataloader import LoRADataset

logger = logging.getLogger(__name__)

class LoRAGeneratorEvaluator:
    """LoRA生成器评估器"""
    
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.device = get_device(config)
        self.processor = LoRAProcessor(config)
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        
        logger.info("Evaluator initialized")
    
    def _load_model(self, checkpoint_path: str):
        """加载训练好的模型"""
        # 创建模型架构
        if self.config.model.model_type == "unet":
            model = UNet1D(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.model_type}")
        
        # 创建DDPM
        ddpm = DDPM(
            timesteps=self.config.model.timesteps,
            model=model,
            beta_schedule=self.config.model.beta_schedule,
            device=self.device
        )
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 处理多GPU训练保存的模型
        if 'ddpm_state' in checkpoint:
            state_dict = checkpoint['ddpm_state']
        else:
            state_dict = checkpoint
        
        # 移除module前缀（如果是多GPU训练保存的）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        ddpm.load_state_dict(new_state_dict)
        ddpm.eval()
        
        logger.info(f"Model loaded from {checkpoint_path}")
        return ddpm
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int, method: str = "ddim") -> List[Dict[str, np.ndarray]]:
        """生成样本"""
        # 确定输入形状
        if self.config.model.use_vae:
            shape = (num_samples, self.config.model.latent_dim)
        else:
            shape = (num_samples, self.processor.total_dim)
        
        # 生成样本
        samples, _ = self.model.sample(shape, method=method)
        
        # 解码样本
        generated_params = []
        for i in range(num_samples):
            if self.config.model.use_vae:
                # 需要加载VAE来解码
                vae = LoRAVAE(self.config).to(self.device)
                # 这里需要加载训练好的VAE权重
                # vae.load_state_dict(...)
                sample_dict = vae.decode(samples[i].unsqueeze(0))
            else:
                sample_dict = self.processor.unflatten_lora(samples[i])
            
            # 转换为numpy
            numpy_dict = {key: tensor.detach().cpu().numpy() for key, tensor in sample_dict.items()}
            generated_params.append(numpy_dict)
        
        return generated_params
    
    def compute_reconstruction_error(self, original_data: List[Dict], generated_data: List[Dict]) -> Dict[str, float]:
        """计算重建误差（如果使用VAE）"""
        errors = {}
        
        for key in ['lora1_a', 'lora1_b', 'lora2_a', 'lora2_b']:
            original_values = []
            generated_values = []
            
            for orig, gen in zip(original_data, generated_data):
                original_values.append(orig[key].flatten())
                generated_values.append(gen[key].flatten())
            
            # 计算MSE
            mse = np.mean([np.mean((orig - gen) ** 2) for orig, gen in zip(original_values, generated_values)])
            errors[f'{key}_mse'] = mse
            
            # 计算MAE
            mae = np.mean([np.mean(np.abs(orig - gen)) for orig, gen in zip(original_values, generated_values)])
            errors[f'{key}_mae'] = mae
        
        return errors
    
    def compute_distribution_similarity(self, original_data: List[Dict], generated_data: List[Dict]) -> Dict[str, float]:
        """计算分布相似性"""
        metrics = {}
        
        for key in ['lora1_a', 'lora1_b', 'lora2_a', 'lora2_b']:
            # 展平所有样本
            orig_flattened = np.concatenate([d[key].flatten() for d in original_data])
            gen_flattened = np.concatenate([d[key].flatten() for d in generated_data])
            
            # 计算统计量
            metrics[f'{key}_mean_diff'] = np.abs(np.mean(orig_flattened) - np.mean(gen_flattened))
            metrics[f'{key}_std_diff'] = np.abs(np.std(orig_flattened) - np.std(gen_flattened))
            metrics[f'{key}_min_diff'] = np.abs(np.min(orig_flattened) - np.min(gen_flattened))
            metrics[f'{key}_max_diff'] = np.abs(np.max(orig_flattened) - np.max(gen_flattened))
        
        return metrics
    
    def evaluate_diversity(self, generated_data: List[Dict]) -> Dict[str, float]:
        """评估生成样本的多样性"""
        diversity_metrics = {}
        
        for key in ['lora1_a', 'lora1_b', 'lora2_a', 'lora2_b']:
            # 计算样本间的平均距离
            distances = []
            samples = [d[key].flatten() for d in generated_data]
            
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    dist = np.linalg.norm(samples[i] - samples[j])
                    distances.append(dist)
            
            if distances:
                diversity_metrics[f'{key}_avg_distance'] = np.mean(distances)
                diversity_metrics[f'{key}_std_distance'] = np.std(distances)
            else:
                diversity_metrics[f'{key}_avg_distance'] = 0
                diversity_metrics[f'{key}_std_distance'] = 0
        
        return diversity_metrics
    
    def comprehensive_evaluation(self, test_data_path: str, num_generated: int = 1000):
        """综合评估"""
        logger.info("Starting comprehensive evaluation...")
        
        # 加载测试数据
        test_dataset = LoRADataset(test_data_path, self.config, max_samples=num_generated)
        test_data = [test_dataset[i] for i in range(min(len(test_dataset), num_generated))]
        
        # 转换为numpy格式
        test_data_numpy = []
        for item in test_data:
            numpy_item = {key: tensor.numpy() for key, tensor in item.items()}
            test_data_numpy.append(numpy_item)
        
        # 生成样本
        logger.info(f"Generating {num_generated} samples...")
        generated_data = self.generate_samples(num_generated)
        
        # 计算各项指标
        results = {}
        
        # 重建误差
        if self.config.model.use_vae:
            recon_errors = self.compute_reconstruction_error(test_data_numpy, generated_data)
            results.update(recon_errors)
        
        # 分布相似性
        dist_similarity = self.compute_distribution_similarity(test_data_numpy, generated_data)
        results.update(dist_similarity)
        
        # 多样性
        diversity = self.evaluate_diversity(generated_data)
        results.update(diversity)
        
        # 打印结果
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        return results, generated_data

def main():
    """主评估函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LoRA Generator")
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--test_data', type=str, required=True, help='Test data path')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config.load(args.config)
    
    # 设置
    set_seed(config.system.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )
    
    # 创建评估器
    evaluator = LoRAGeneratorEvaluator(config, args.checkpoint)
    
    # 执行评估
    results, generated_data = evaluator.comprehensive_evaluation(
        args.test_data, 
        args.num_samples
    )
    
    # 保存结果
    import json
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存生成的样本
    with open(output_dir / 'generated_samples.pkl', 'wb') as f:
        pickle.dump(generated_data, f)
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()