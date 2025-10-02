import torch
import numpy as np
import random
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pickle

def setup_logging(log_dir: str, level: int = logging.INFO):
    """设置日志配置"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

def set_seed(seed: int = 42, deterministic: bool = True):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(config) -> torch.device:
    """根据配置获取设备"""
    if hasattr(config, 'system'):
        device_config = config.system
    else:
        device_config = config
    
    if device_config.device == "cuda" and torch.cuda.is_available():
        if hasattr(device_config, 'gpu_ids') and device_config.gpu_ids:
            device = torch.device(f"cuda:{device_config.gpu_ids[0]}")
            print(f"Using GPU: {torch.cuda.get_device_name(device_config.gpu_ids[0])}")
        else:
            device = torch.device("cuda:0")
            print("Using default GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def params_to_vec(params: List[Union[np.ndarray, torch.Tensor]], device: str = 'cpu') -> torch.Tensor:
    """
    将LoRA权重矩阵列表展平并拼接成一个向量
    """
    vecs = []
    for p in params:
        if isinstance(p, np.ndarray):
            vec = torch.from_numpy(p).float().flatten()
        else:
            vec = p.float().flatten()
        vecs.append(vec)
    
    return torch.cat(vecs).to(device)

def vec_to_params(vec: torch.Tensor, param_dims: List) -> List[torch.Tensor]:
    """
    将一维向量还原为原始的LoRA权重矩阵列表
    """
    vec = vec.cpu()
    params = []
    current_pos = 0
    
    for dim in param_dims:
        num_elements = dim[0] * dim[1]
        param = vec[current_pos: current_pos + num_elements].reshape(dim)
        params.append(param)
        current_pos += num_elements
    
    return params

def save_generated_params(params_list: List, path: str):
    """
    将生成的权重列表保存为pkl文件
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(params_list, f)
    
    print(f"Generated parameters saved to {path}")

def prepare_data_for_model(data, config, device):
    """
    将一批原始数据转换为模型所需的张量格式
    """
    from data.processor import LoRAProcessor
    
    processor = LoRAProcessor(config)
    
    # 将每个样本（4个矩阵）转换为一个扁平向量
    batch_vecs = [processor.flatten_lora(p) for p in data]
    
    # 堆叠成一个 batch 张量
    batch_tensor = torch.stack(batch_vecs)
    
    # 增加一个 channel 维度，以适配 Conv1d 输入 (N, C, L)
    return batch_tensor.unsqueeze(1)

def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: torch.nn.Module):
    """打印模型摘要"""
    total_params = 0
    print("Model Summary:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Params':>20} {'Shape':>20}")
    print("-" * 80)
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
        print(f"{name:<40} {params:>20,} {str(list(parameter.shape)):>20}")
    
    print("-" * 80)
    print(f"{'Total parameters':<40} {total_params:>20,}")
    print("-" * 80)
    
    return total_params

def save_checkpoint(
    state: Dict, 
    filename: str, 
    is_best: bool = False, 
    best_filename: str = 'best_model.pth'
):
    """保存检查点"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def load_checkpoint(filename: str, device: torch.device) -> Dict:
    """加载检查点"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    checkpoint = torch.load(filename, map_location=device)
    return checkpoint

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """计算梯度范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def compute_parameter_norm(model: torch.nn.Module) -> float:
    """计算参数范数"""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """创建实验目录"""
    import datetime
    
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, experiment_name)
    
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    subdirs = ['checkpoints', 'logs', 'samples', 'configs']
    for subdir in subdirs:
        Path(os.path.join(exp_dir, subdir)).mkdir(parents=True, exist_ok=True)
    
    return exp_dir

def save_config(config: Any, path: str):
    """保存配置到文件"""
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

def load_config(path: str) -> Dict:
    """从文件加载配置"""
    with open(path, 'r') as f:
        return json.load(f)

def compute_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                           mu2: np.ndarray, sigma2: np.ndarray, 
                           eps: float = 1e-6) -> float:
    """计算Frechet距离（用于评估生成质量）"""
    try:
        # 计算均值差的平方
        diff = mu1 - mu2
        diff_squared = np.dot(diff, diff)
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # 计算Frechet距离
        fid = diff_squared + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    except Exception as e:
        logging.warning(f"Error computing FID: {e}")
        return float('inf')

class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._data = {}
        self._counts = {}
    
    def update(self, metrics: Dict[str, float], n: int = 1):
        for key, value in metrics.items():
            if key not in self._data:
                self._data[key] = 0.0
                self._counts[key] = 0
            self._data[key] += value * n
            self._counts[key] += n
    
    def average(self) -> Dict[str, float]:
        return {key: self._data[key] / self._counts[key] for key in self._data}
    
    def get(self, key: str) -> float:
        return self._data[key] / self._counts[key]