import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pickle


class LoRAProcessor:
    """LoRA数据处理器 - 优化版本"""
    
    def __init__(self, config):
        self.config = config
        self.shapes = config.model.lora_shapes
        
        # 计算各矩阵维度
        self.dims = {key: np.prod(shape) for key, shape in self.shapes.items()}
        self.total_dim = sum(self.dims.values())
        
        # 索引映射
        self.indices = {}
        start_idx = 0
        for key in ['lora1_a', 'lora1_b', 'lora2_a', 'lora2_b']:
            size = self.dims[key]
            self.indices[key] = (start_idx, start_idx + size)
            start_idx += size
        
        # 统计信息
        self.stats = None
    
    def compute_statistics(self, data_loader):
        """从数据加载器计算统计信息"""
        all_values = []
        for batch in data_loader:
            flat = self.flatten_lora(batch)
            all_values.append(flat)
        
        all_tensor = torch.cat(all_values, dim=0)
        self.stats = {
            'mean': all_tensor.mean(dim=0),
            'std': all_tensor.std(dim=0),
            'min': all_tensor.min(dim=0)[0],
            'max': all_tensor.max(dim=0)[0]
        }
        return self.stats
    
    def flatten_lora(self, lora_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将LoRA矩阵展平为向量"""
        flattened = []
        for key in ['lora1_a', 'lora1_b', 'lora2_a', 'lora2_b']:
            tensor = lora_dict[key]
            if tensor.dim() > 2:  # 批量数据
                flattened.append(tensor.reshape(tensor.shape[0], -1))
            else:
                flattened.append(tensor.flatten().unsqueeze(0))
        
        return torch.cat(flattened, dim=-1)
    
    def unflatten_lora(self, flat_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将展平的向量恢复为LoRA矩阵"""
        result = {}
        for key in ['lora1_a', 'lora1_b', 'lora2_a', 'lora2_b']:
            start, end = self.indices[key]
            shape = self.shapes[key]
            
            if flat_vector.dim() > 1:  # 批量数据
                result[key] = flat_vector[:, start:end].reshape(-1, *shape)
            else:
                result[key] = flat_vector[start:end].reshape(shape)
        
        return result
    
    def normalize(self, lora_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """归一化LoRA矩阵"""
        if self.config.data.use_minmax_scaler:
            # Min-Max归一化到[-1, 1]
            return {
                key: 2 * (tensor - self.stats['min'][self.indices[key][0]:self.indices[key][1]].mean()) / 
                     (self.stats['max'][self.indices[key][0]:self.indices[key][1]].mean() - 
                      self.stats['min'][self.indices[key][0]:self.indices[key][1]].mean() + 1e-8) - 1
                for key, tensor in lora_dict.items()
            }
        else:
            # Z-score标准化
            return {
                key: (tensor - self.stats['mean'][self.indices[key][0]:self.indices[key][1]].mean()) / 
                     (self.stats['std'][self.indices[key][0]:self.indices[key][1]].mean() + 1e-8)
                for key, tensor in lora_dict.items()
            }
    
    def denormalize(self, lora_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """反归一化LoRA矩阵"""
        if self.config.data.use_minmax_scaler:
            # 从[-1, 1]恢复
            return {
                key: (tensor + 1) / 2 * (self.stats['max'][self.indices[key][0]:self.indices[key][1]].mean() - 
                     self.stats['min'][self.indices[key][0]:self.indices[key][1]].mean()) + 
                     self.stats['min'][self.indices[key][0]:self.indices[key][1]].mean()
                for key, tensor in lora_dict.items()
            }
        else:
            # Z-score恢复
            return {
                key: tensor * self.stats['std'][self.indices[key][0]:self.indices[key][1]].mean() + 
                     self.stats['mean'][self.indices[key][0]:self.indices[key][1]].mean()
                for key, tensor in lora_dict.items()
            }

class MatrixDecomposer:
    """矩阵分解工具 - 用于降维"""
    
    def __init__(self, method: str = "pca"):
        self.method = method
        self.components_ = None
        
    def fit(self, matrices: List[torch.Tensor], target_dim: int):
        """训练分解器"""
        if self.method == "pca":
            return self._fit_pca(matrices, target_dim)
        elif self.method == "ae":
            return self._fit_ae(matrices, target_dim)
    
    def _fit_pca(self, matrices: List[torch.Tensor], target_dim: int):
        """PCA分解"""
        from sklearn.decomposition import PCA
        
        # 展平矩阵
        flattened = [mat.flatten().numpy() for mat in matrices]
        flattened = np.array(flattened)
        
        self.pca = PCA(n_components=target_dim)
        self.pca.fit(flattened)
        self.components_ = torch.from_numpy(self.pca.components_).float()
        
    def transform(self, matrix: torch.Tensor) -> torch.Tensor:
        """转换矩阵到低维空间"""
        if self.method == "pca":
            flattened = matrix.flatten().numpy()
            return torch.from_numpy(self.pca.transform([flattened])[0]).float()
    
    def inverse_transform(self, code: torch.Tensor) -> torch.Tensor:
        """从低维空间重建矩阵"""
        if self.method == "pca":
            reconstructed = self.pca.inverse_transform(code.numpy())
            original_shape = self.original_shape
            return torch.from_numpy(reconstructed).float().reshape(original_shape)