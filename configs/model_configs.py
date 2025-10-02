from configs.config_base import Config, ModelConfig, TrainConfig

def get_small_config():
    """小型配置（测试用）"""
    config = Config()
    config.model.latent_dim = 128
    config.model.unet_dim = 64
    config.model.transformer_dim = 256
    config.train.batch_size = 16
    config.train.num_epochs = 100
    return config

def get_medium_config():
    """中型配置"""
    config = Config()
    config.model.latent_dim = 256
    config.model.unet_dim = 128
    config.model.transformer_dim = 512
    config.train.batch_size = 32
    return config

def get_large_config():
    """大型配置"""
    config = Config()
    config.model.latent_dim = 512
    config.model.unet_dim = 256  
    config.model.transformer_dim = 768
    config.model.vae_hidden_dims = [2048, 1024, 512]
    config.train.batch_size = 16
    return config

def get_transformer_config():
    """Transformer专用配置"""
    config = Config()
    config.model.model_type = "transformer"
    config.model.latent_dim = 512
    config.train.batch_size = 16
    return config

def get_fast_config():
    """快速训练配置"""
    config = Config()
    config.model.timesteps = 500
    config.model.use_vae = False
    config.train.batch_size = 64
    config.train.num_epochs = 200
    return config