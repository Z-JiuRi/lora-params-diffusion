# 导入必要的库
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any

# 使用dataclass装饰器定义模型配置类
@dataclass
class ModelConfig:
    """模型配置"""
    # 基础配置
    model_type: Literal["unet", "transformer", "mlp"] = "transformer"  # 模型类型选择，默认为UNet
    
    # LoRA维度配置
    lora_shapes: Dict[str, tuple] = field(default_factory=lambda: {
        'lora1_a': (16, 64),    # LoRA层1的A矩阵形状
        'lora1_b': (2048, 16),  # LoRA层1的B矩阵形状
        'lora2_a': (16, 2048),  # LoRA层2的A矩阵形状
        'lora2_b': (64, 16)     # LoRA层2的B矩阵形状
    })
    
    # 生成策略选择
    generation_strategy: Literal["joint", "separate", "hierarchical"] = "joint"
    
    # 潜在空间配置
    latent_dim: int = 256        # 潜在空间维度大小
    use_vae: bool = True        # 是否使用VAE
    vae_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])  # VAE隐藏层维度
    vae_beta: float = 0.01      # VAE中KL散度的权重系数
    
    # 扩散模型配置
    timesteps: int = 1000       # 扩散过程的时间步数
    beta_schedule: Literal["linear", "cosine", "quadratic", "sigmoid"] = "cosine"  # β调度策略
    loss_type: Literal["l1", "l2", "huber"] = "l2"  # 损失函数类型
    
    # UNet配置
    unet_dim: int = 128                         # UNet基础维度
    unet_dim_mults: tuple = (1, 2, 4, 8)        # 各层维度倍增系数
    unet_attention_resolutions: tuple = (16,)    # 注意力分辨率
    unet_heads: int = 8                         # 注意力头数
    unet_dropout: float = 0.1                   # Dropout率
    
    # Transformer配置
    transformer_dim: int = 512                  # Transformer隐藏层维度
    transformer_depth: int = 12                 # Transformer层数
    transformer_heads: int = 8                  # 注意力头数
    transformer_mlp_dim: int = 2048              # MLP中间层维度
    transformer_dropout: float = 0.1             # Dropout率
    
    # MLP配置（简单基线模型）
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])  # MLP隐藏层维度

@dataclass
class TrainConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 4            # 批量大小
    learning_rate: float = 1e-4     # 学习率
    num_epochs: int = 1000          # 训练轮数
    gradient_clip: float = 1.0      # 梯度裁剪阈值
    weight_decay: float = 0.01      # 权重衰减系数
    exp_name: str = "default"       # 实验名称
    
    # 优化器配置
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"  # 优化器类型
    adam_beta1: float = 0.9         # Adam优化器的beta1参数
    adam_beta2: float = 0.999       # Adam优化器的beta2参数
    adam_epsilon: float = 1e-8      # Adam优化器的epsilon参数
    
    # 学习率调度
    scheduler: Literal["none", "cosine", "linear", "exponential"] = "cosine"  # 学习率调度器类型
    warmup_steps: int = 1000        # 预热步数
    lr_decay: float = 0.95          # 学习率衰减率
    
    # EMA配置（指数移动平均）
    use_ema: bool = True            # 是否使用EMA
    ema_decay: float = 0.9999       # EMA衰减率
    
    # 条件生成配置
    conditional: bool = False        # 是否使用条件生成
    condition_dim: int = 512         # 条件维度大小
    
    # 早停配置
    early_stopping: bool = True     # 是否启用早停
    patience: int = 20              # 早停耐心值（连续多少个epoch不提升后停止）

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data/params"        # 数据目录路径
    train_split: float = 0.9        # 训练集比例
    val_split: float = 0.1          # 验证集比例
    test_split: float = 0.0         # 测试集比例
    
    # 预处理配置
    normalize: bool = True          # 是否归一化
    standardize: bool = False       # 是否标准化
    use_minmax_scaler: bool = True  # 是否使用MinMax缩放
    
    # 缓存配置
    cache_data: bool = True         # 是否缓存数据
    cache_dir: str = "./cache"      # 缓存目录路径

@dataclass 
class SystemConfig:
    """系统配置"""
    device: Literal["cuda", "cpu", "auto"] = "cpu"  # 运行设备，auto自动选择
    gpu_ids: List[int] = field(default_factory=lambda: [0])  # 使用的GPU ID列表
    num_workers: int = 4            # 数据加载器的工作线程数
    use_multi_gpu: bool = False     # 是否使用多GPU
    pin_memory: bool = True         # 是否将数据加载到CUDA固定内存中
    distributed: bool = False       # 是否使用分布式训练
    
    # 性能优化配置
    mixed_precision: bool = True    # 是否使用混合精度训练
    gradient_checkpointing: bool = False  # 是否使用梯度检查点技术
    
    # 随机种子配置
    seed: int = 42                  # 随机种子
    deterministic: bool = True       # 是否启用确定性模式

@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)             # 模型配置实例
    train: TrainConfig = field(default_factory=TrainConfig)             # 训练配置实例
    system: SystemConfig = field(default_factory=SystemConfig)          # 系统配置实例
    data: DataConfig = field(default_factory=DataConfig)                # 数据配置实例
    
    def __post_init__(self):
        """初始化后处理"""
        if self.system.device == "auto":
            # 自动检测并设置设备（优先使用CUDA）
            self.system.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 多GPU配置检查
        if self.system.use_multi_gpu and self.system.device == "cuda":
            if torch.cuda.device_count() > 1:
                if not self.system.gpu_ids:
                    # 如果没有指定GPU ID，则使用所有可用的GPU
                    self.system.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                # 如果只有1个GPU，则禁用多GPU设置
                self.system.use_multi_gpu = False
                self.system.gpu_ids = [0]
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        import json
        from dataclasses import asdict
        with open(path, 'w') as f:
            # 将配置对象转换为字典并保存为JSON
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """从JSON文件加载配置"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        # 从字典创建配置对象
        return cls(
            model=ModelConfig(**data.get('model', {})),
            train=TrainConfig(**data.get('train', {})),
            system=SystemConfig(**data.get('system', {})),
            data=DataConfig(**data.get('data', {}))
        )