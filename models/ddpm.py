import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Beta调度函数
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    余弦调度，参考论文 https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """线性beta调度"""
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """二次beta调度"""
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """S型beta调度"""
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_beta_schedule(schedule_name: str, timesteps: int, **kwargs) -> torch.Tensor:
    """获取beta调度"""
    if schedule_name == "cosine":
        return cosine_beta_schedule(timesteps, **kwargs)
    elif schedule_name == "linear":
        return linear_beta_schedule(timesteps, **kwargs)
    elif schedule_name == "quadratic":
        return quadratic_beta_schedule(timesteps, **kwargs)
    elif schedule_name == "sigmoid":
        return sigmoid_beta_schedule(timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_name}")

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Models (DDPM)
    优化版本，支持多种调度和配置
    """
    
    def __init__(
        self, 
        timesteps: int = 1000,
        model: Optional[nn.Module] = None,
        beta_schedule: str = "cosine",
        loss_type: str = "l2",
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        
        self.timesteps = timesteps
        self.model = model
        self.loss_type = loss_type
        self.device = device
        
        # 定义beta调度
        self.betas = get_beta_schedule(beta_schedule, timesteps).to(device)
        
        # 计算alpha相关参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 扩散过程相关参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # 后验分布相关参数
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # 记录调度信息
        logger.info(f"Initialized DDPM with {timesteps} timesteps, {beta_schedule} schedule")
    
    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        从参数a中提取与时间步t对应的值，并调整形状以适配输入x
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散过程：根据扩散公式生成噪声样本
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        从噪声预测原始样本
        """
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return sqrt_recip_alphas_t * (x_t - sqrt_one_minus_alphas_cumprod_t * noise)
    
    def p_losses(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        loss_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        计算模型损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        if loss_type is None:
            loss_type = self.loss_type
        
        # 添加噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 预测噪声
        predicted_noise = self.model(x_noisy, t, condition)
        
        # 计算损失
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        condition: Optional[torch.Tensor] = None,
        t_index: int = 0,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        单步反向采样
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # 预测噪声
        predicted_noise = self.model(x, t, condition)
        
        # 分类器自由引导（如果使用条件生成）
        if condition is not None and guidance_scale != 1.0:
            uncond_predicted_noise = self.model(x, t, None)
            predicted_noise = uncond_predicted_noise + guidance_scale * (predicted_noise - uncond_predicted_noise)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self, 
        shape: Tuple[int, ...], 
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        progress: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样主循环，从纯噪声逐步生成样本
        """
        device = next(self.model.parameters()).device
        
        batch_size = shape[0]
        # 初始化为纯噪声
        x = torch.randn(shape, device=device)
        
        # 存储所有中间样本（用于可视化）
        all_samples = []
        
        timesteps = list(reversed(range(0, self.timesteps)))
        if progress:
            timesteps = tqdm(timesteps, desc='Sampling loop')
        
        for i in timesteps:
            x = self.p_sample(
                x, 
                torch.full((batch_size,), i, device=device, dtype=torch.long), 
                condition, 
                i,
                guidance_scale
            )
            all_samples.append(x.cpu())
        
        all_samples = torch.stack(all_samples, dim=0)
        return x, all_samples
    
    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        sampling_timesteps: int = 50,
        eta: float = 0.0,
        progress: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM采样，加速采样过程
        """
        device = next(self.model.parameters()).device
        
        # 创建采样时间步
        times = torch.linspace(-1, self.timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x = torch.randn(shape, device=device)
        all_samples = [x.cpu()]
        
        if progress:
            time_pairs = tqdm(time_pairs, desc='DDIM sampling')
        
        for time, time_next in time_pairs:
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next] if time_next >= 0 else torch.tensor(1.0)
            
            # 预测噪声
            t = torch.full((shape[0],), time, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t, condition)
            
            # 预测原始样本
            x0 = self.predict_start_from_noise(x, t, predicted_noise)
            
            # 计算方向
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
            c = torch.sqrt(1 - alpha_next - sigma ** 2)
            
            noise = torch.randn_like(x) if time_next > 0 else 0
            
            x = torch.sqrt(alpha_next) * x0 + c * predicted_noise + sigma * noise
            all_samples.append(x.cpu())
        
        all_samples = torch.stack(all_samples, dim=0)
        return x, all_samples
    
    @torch.no_grad()
    def sample(
        self, 
        shape: Tuple[int, ...], 
        condition: Optional[torch.Tensor] = None,
        method: str = "ddpm",
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样接口，返回最终样本和采样过程
        """
        if method == "ddpm":
            return self.p_sample_loop(shape, condition, **kwargs)
        elif method == "ddim":
            return self.ddim_sample(shape, condition, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def forward(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        loss_type: Optional[str] = None
    ) -> torch.Tensor:
        return self.p_losses(x_start, t, condition, noise, loss_type)
    
    def save(self, path: str):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str, strict: bool = True):
        """加载模型参数"""
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=False)
            self.load_state_dict(state_dict, strict=strict)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise