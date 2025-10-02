使用说明
1. 安装依赖
bash
pip install -r requirements.txt
2. 训练VAE（如果使用降维）
bash
python scripts/train_vae.py
3. 训练扩散模型
bash
python scripts/train.py
4. 生成样本
bash
python scripts/generate.py
5. 评估模型
bash
python scripts/evaluate.py --config config.json --checkpoint best_model.pt --test_data test.pkl
关键特性
灵活的配置系统：支持多种模型架构和训练配置

多种降维策略：VAE、PCA等

多种扩散模型：DDPM、DDIM等

完整的训练流程：包含早停、EMA、混合精度等

全面的评估指标：重建误差、分布相似性、多样性等

模块化设计：易于扩展和维护