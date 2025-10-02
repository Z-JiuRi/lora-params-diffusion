import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LoRADataset(Dataset):
    """LoRA参数数据集 - 优化版本"""
    
    def __init__(
        self,
        data_path: str,
        config,
        transform=None,
        cache_data: bool = True,
        max_samples: Optional[int] = None
    ):
        self.data_path = data_path
        self.config = config
        self.transform = transform
        self.cache_data = cache_data
        self.max_samples = max_samples
        
        # 加载数据
        self.data = self._load_data()
        
        # 数据统计
        self.compute_statistics()
        
        logger.info(f"Loaded dataset with {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, torch.Tensor]]:
        """加载pkl数据"""
        try:
            with open(self.data_path, 'rb') as f:
                raw_data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise
        
        # 限制样本数量
        if self.max_samples is not None:
            raw_data = raw_data[:self.max_samples]
        
        # 转换为tensor列表
        processed_data = []
        for i, item in enumerate(raw_data):
            try:
                if isinstance(item, dict):
                    # 假设数据已经按字典格式组织
                    processed_item = {
                        'lora1_a': torch.FloatTensor(item['lora1_a']),
                        'lora1_b': torch.FloatTensor(item['lora1_b']),
                        'lora2_a': torch.FloatTensor(item['lora2_a']),
                        'lora2_b': torch.FloatTensor(item['lora2_b'])
                    }
                elif isinstance(item, (list, tuple)) and len(item) == 4:
                    # 假设是四个矩阵的元组或列表
                    processed_item = {
                        'lora1_a': torch.FloatTensor(item[0]),
                        'lora1_b': torch.FloatTensor(item[1]),
                        'lora2_a': torch.FloatTensor(item[2]),
                        'lora2_b': torch.FloatTensor(item[3])
                    }
                else:
                    logger.warning(f"Unexpected data format at index {i}, skipping")
                    continue
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}, skipping")
                continue
        
        return processed_data
    
    def compute_statistics(self):
        """计算数据统计信息"""
        if not self.data:
            self.mean = 0
            self.std = 1
            self.min_val = 0
            self.max_val = 1
            return
        
        all_values = []
        for item in self.data:
            for key in item:
                all_values.append(item[key].flatten())
        
        if all_values:
            all_values = torch.cat(all_values)
            self.mean = all_values.mean().item()
            self.std = all_values.std().item()
            self.min_val = all_values.min().item()
            self.max_val = all_values.max().item()
        else:
            self.mean = 0
            self.std = 1
            self.min_val = 0
            self.max_val = 1
        
        logger.info(f"Data statistics - Mean: {self.mean:.4f}, Std: {self.std:.4f}, "
                   f"Range: [{self.min_val:.4f}, {self.max_val:.4f}]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
        
        return item

class DataAugmentation:
    """数据增强类"""
    
    def __init__(self, config):
        self.config = config
        self.noise_level = getattr(config.train, 'noise_level', 0.01)
        self.scaling_factor = getattr(config.train, 'scaling_factor', 0.1)
    
    def __call__(self, lora_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用数据增强"""
        augmented = {}
        
        for key, tensor in lora_dict.items():
            # 添加高斯噪声
            if self.noise_level > 0:
                noise = torch.randn_like(tensor) * self.noise_level
                tensor = tensor + noise
            
            # 随机缩放
            if self.scaling_factor > 0:
                scale = 1 + (torch.rand(1) * 2 - 1) * self.scaling_factor
                tensor = tensor * scale.item()
            
            augmented[key] = tensor
        
        return augmented


def create_dataloader(config, data_path: str, is_train: bool = True):
    """创建数据加载器"""
    logger.info(f"Creating {'training' if is_train else 'validation'} dataloader for {data_path}")
    
    # 数据增强
    transform = None
    if is_train and getattr(config.train, 'augmentation', False):
        transform = DataAugmentation(config)
    
    # 创建数据集
    dataset = LoRADataset(
        data_path=data_path,
        config=config,
        transform=transform,
        cache_data=config.data.cache_data,
        max_samples=getattr(config.data, 'max_samples', None)
    )
    
    # 设置DataLoader参数
    loader_kwargs = {
        'batch_size': config.train.batch_size,
        'shuffle': is_train,
        'num_workers': config.system.num_workers,
        'pin_memory': config.system.pin_memory and config.system.device == 'cuda',
        'drop_last': is_train
    }
    
    # 如果使用多GPU，调整batch size
    if config.system.use_multi_gpu:
        loader_kwargs['batch_size'] = config.train.batch_size * len(config.system.gpu_ids)
    
    return DataLoader(dataset, **loader_kwargs)

def create_train_val_dataloaders(config):
    """创建训练集和验证集数据加载器"""
    logger.info("Creating train and validation dataloaders")
    
    # 获取数据文件路径
    data_dir = config.data.data_dir
    data_files = []
    
    # 如果是目录，获取所有pkl文件
    if os.path.isdir(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.pkl'):
                data_files.append(os.path.join(data_dir, file))
    # 如果是单个文件
    elif os.path.isfile(data_dir) and data_dir.endswith('.pkl'):
        data_files = [data_dir]
    else:
        raise ValueError(f"无效的数据路径: {data_dir}")
    
    if not data_files:
        raise ValueError(f"No .pkl files found in {data_dir}")
    
    logger.info(f"Found {len(data_files)} data files")
    
    # 加载所有数据
    all_data = []
    for file_path in data_files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                all_data.extend(data)
                logger.info(f"Loaded {len(data)} samples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data loaded from any files")
    
    logger.info(f"Total samples: {len(all_data)}")
    
    # # 随机打乱数据
    # np.random.seed(config.system.seed)
    # indices = np.random.permutation(len(all_data))
    # # 根据验证集比例分割数据
    # val_split = getattr(config.data, 'val_split', 0.1)
    # val_size = int(len(all_data) * val_split)
    # train_indices = indices[val_size:]
    # val_indices = indices[:val_size]
    
    val_split = getattr(config.data, 'val_split', 0.1)
    tot_size = len(all_data)
    val_size = int(tot_size * val_split)
    
    # 分割数据
    train_data = [all_data[i] for i in range(tot_size - val_size)]
    val_data = [all_data[i] for i in range(tot_size - val_size, tot_size)]
    
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # 保存临时文件
    temp_dir = os.path.join(os.path.dirname(data_dir), 'temp_split')
    os.makedirs(temp_dir, exist_ok=True)
    
    train_path = os.path.join(temp_dir, 'train_data.pkl')
    val_path = os.path.join(temp_dir, 'val_data.pkl')
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    
    logger.info(f"Saved split data to {temp_dir}")
    
    # 创建数据加载器
    train_loader = create_dataloader(config, train_path, is_train=True)
    val_loader = create_dataloader(config, val_path, is_train=False)
    
    return train_loader, val_loader

def collate_lora_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """整理批次数据"""
    if not batch:
        return {}
    
    collated = {}
    for key in batch[0].keys():
        try:
            collated[key] = torch.stack([item[key] for item in batch])
        except Exception as e:
            logger.error(f"Error collating key {key}: {e}")
            # 创建一个默认的张量
            example_tensor = batch[0][key]
            collated[key] = torch.zeros(len(batch), *example_tensor.shape)
    
    return collated