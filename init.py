"""
LoRA Diffusion - A diffusion model for generating LoRA parameters
"""

__version__ = "1.0.0"
__author__ = "z-jiuri"
__email__ = "email@example.com"

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)
else:
    print(f"Project root {project_root} in sys.path")
from configs.config_base import Config
from configs.model_configs import (
    get_small_config, 
    get_medium_config, 
    get_large_config,
    get_transformer_config,
    get_fast_config
)

# 导出主要类
__all__ = [
    "Config",
    "get_small_config",
    "get_medium_config", 
    "get_large_config",
    "get_transformer_config",
    "get_fast_config"
]