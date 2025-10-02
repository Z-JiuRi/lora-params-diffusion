import torch
import pickle
import os
from pathlib import Path

from configs.config_base import Config
from models.ddpm import DDPM
from models.unet import UNet1D
from models.vae import LoRAVAE
from data.processor import LoRAProcessor
from utils.utils import set_seed

def generate_samples(config_path: str = None, num_samples: int = 10, output_path: str = "./generated"):
    """生成LoRA参数样本"""
    
    # 加载配置
    if config_path:
        config = Config.load(config_path)
    else:
        from configs.model_configs import get_medium_config
        config = get_medium_config()
    
    set_seed(config.system.seed)
    device = torch.device(config.system.device)
    
    # 创建模型
    processor = LoRAProcessor(config)
    
    if config.model.model_type == "unet":
        model = UNet1D(config)
    else:
        raise ValueError("Only UNet supported for generation")
    
    ddpm = DDPM(config.model.timesteps, model).to(device)
    
    # VAE（如果使用）
    vae = None
    if config.model.use_vae:
        vae = LoRAVAE(config).to(device)
    
    # 加载检查点
    checkpoint_path = "./best_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(checkpoint['ddpm_state'])
    if vae and 'vae_state' in checkpoint:
        vae.load_state_dict(checkpoint['vae_state'])
    
    ddpm.eval()
    if vae:
        vae.eval()
    
    # 生成样本
    print(f"Generating {num_samples} samples...")
    
    if config.model.use_vae:
        shape = (num_samples, config.model.latent_dim)
    else:
        shape = (num_samples, processor.total_dim)
    
    with torch.no_grad():
        samples, _ = ddpm.sample(shape)
    
    # 解码样本
    generated_params = []
    for i in range(num_samples):
        if vae:
            sample_dict = vae.decode(samples[i].unsqueeze(0))
        else:
            sample_dict = processor.unflatten_lora(samples[i])
        
        # 转换为numpy并保存
        numpy_dict = {key: tensor.cpu().numpy() for key, tensor in sample_dict.items()}
        generated_params.append(numpy_dict)
    
    # 保存结果
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_path, "generated_lora_params.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(generated_params, f)
    
    print(f"Generated {num_samples} samples saved to {output_file}")
    return generated_params

if __name__ == "__main__":
    generate_samples()