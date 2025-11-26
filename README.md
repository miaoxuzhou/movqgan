# MoVQGAN: 

调整了 MoVQGAN 原代码(https://github.com/ai-forever/MoVQGAN)的结构，主要调整如下：

1. **模块化损失函数**：独立计算每种损失.
2. **设计VQ模型基类**：原代码中的 VQ、MoVQ 存在大量重复代码，提供了统一的基类.

## 📋 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [训练](#训练)

## 🚀 安装

### 环境要求

基本与原论文一致, 但原论文使用的 pytorch-ligtning 版本过早, 与 torch 等其他库可能不兼容, 因此, 将 pytorch-lightning 的版本修改为 2.1.0, 其余库的版本使用最新版本即可，或参照 requirements.txt 文件.

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/miaoxuzhou/movqgan.git
cd movqgan

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 主要依赖包

```
torch
torchvision
pytorch_lightning
omegaconf
einops
fsspec
wandb
transformers
```

## 快速开始

```bash
# 训练 67M 模型
python main.py --config configs/movqgan_67M.yaml

# 训练 102M 模型
python main.py --config configs/movqgan_102M.yaml

# 训练 270M 模型
python main.py --config configs/movqgan_270M.yaml
```

## 🏗️ 模型架构

MoVQGAN 由三个主要组件构成：

### 1. 编码器（Encoder）
- 通过 ResNet 块对输入图像进行下采样
- 在指定分辨率应用自注意力机制
- 输出连续的潜在表示

### 2. 向量量化器（Vector Quantizer）
- 使用可学习的码本将连续潜在表示离散化
- 码本大小为 16,384 个条目, 嵌入维度为 4

### 3. MoVQ 解码器（MoVQ Decoder）
- **空间归一化**：使用量化码对特征进行调制
- 通过空间调制的 ResNet 块对潜在表示进行上采样
- 应用带空间调制的注意力机制
- 重建高质量图像

## 📁 项目结构

```
movqgan/
├── configs/                  # 配置文件
│   ├── movqgan_67M.yaml      # 67M 模型配置
│   ├── movqgan_102M.yaml     # 102M 模型配置
│   └── movqgan_270M.yaml     # 270M 模型配置
├── movqgan/
│   ├── data/
│   │   └── dataset.py        # 数据加载器
│   ├── losses/
│   │   ├── adversarial.py    # GAN 损失
│   │   ├── perceptual.py     # LPIPS 感知损失
│   │   └── vqgan_loss.py     # 组合损失
│   ├── models/
│   │   ├── base_vq.py        # 基础 VQ 模型
│   │   ├── vq.py             # 标准 VQ 模型
│   │   ├── movq.py           # 带空间归一化的 MoVQ 模型
│   │   ├── gumbel_vq.py      # Gumbel-Softmax VQ
│   │   └── ema.py            # 指数移动平均
│   ├── modules/
│   │   ├── components/       # 构建模块
│   │   │   ├── attention.py  # 注意力机制
│   │   │   ├── normalization.py  # 空间归一化
│   │   │   ├── residual.py   # ResNet 块
│   │   │   └── sampling.py   # 上/下采样
│   │   ├── encoders/
│   │   │   └── encoder.py       # 图像编码器
│   │   ├── decoders/
│   │   │   ├── decoder.py       # 标准解码器
│   │   │   └── movq_decoder.py  # MoVQ 解码器
│   │   ├── discriminator/
│   │   │   └── discriminator.py  # PatchGAN 判别器
│   │   └── quantizers/
│   │       └── vector_quantizer.py  # VQ 层
│   └── util.py               # 工具函数
├── main.py                   # 训练脚本
├── requirements.txt          # 依赖项
└── setup.py                  # 包设置
└── README.md
```

## 🎓 训练

### 数据集准备

准备一个包含图像路径的 CSV 文件：

```csv
image_name
/path/to/image1.jpg
/path/to/image2.jpg
/path/to/image3.jpg
```

### 配置文件

编辑 YAML 配置文件（例如 `configs/movqgan_67M.yaml`）：

```yaml
# 检查点路径（留空表示从头训练）
ckpt_path: ''

# Weights & Biases 配置
wandb_entity_name: 'your_entity'
wandb_project_name: 'movqgan-67M'

# 训练设置
trainer:
  devices: 4                    # GPU 数量，原论文的 GPU 数量为 4
  num_nodes: 1                  # 节点数量
  accelerator: 'gpu'
  precision: 16                 # 混合精度训练，原论文设置为32
  max_steps: 9999999            # 最大训练步数，根据 bacth_size 和 期望的 epoch 数计算得来
  log_every_n_steps: 10         # 每 10 步记录一次日志
  strategy: 'ddp_find_unused_parameters_true'   # 分布式

# 模型检查点路径设置
ModelCheckpoint:
  dirpath: './checkpoints/movqgan_67M'    # 自定义检查点路径
  filename: "step_{step:07d}"             # 检查点文件名
  save_top_k: -1                          # 保存所有检查点
  every_n_train_steps: 5000
  save_last: true
  
# 数据配置
data:
  train:
    df_path: ./dataset.csv     # 数据集 CSV 路径
    image_size: 256            # 图像分辨率
    batch_size: 4              # 每个 GPU 的批次大小，原论文批次大小为 48
    num_workers: 12            # 数据加载线程数

# 模型参数配置
model:
  target: movqgan.models.movq.MOVQ    # 模型类的路径
  params:
    learning_rate: 0.0001
    ema_decay: 0.9999           # EMA 衰减率(Exponential moving average)
    embed_dim: 4                # 码本嵌入维度
    n_embed: 16384              # 码本大小          
    monitor: val/rec_loss       # 验证指标
    
    # 编码器 / 解码器配置
    ddconfig:
      double_z: false                 # 不输出 mean 和 variance
      z_channels: 4                   # 连续编码维度
      resolution: 256                 # 输入图片分辨率
      in_channels: 3                  # 输入图片通道数
      out_ch: 3                       # 输出通道数
      ch: 128                         # 基础通道数
      ch_mult: [1, 2, 2, 4]           # 解码 / 编码时通道数的倍数
      num_res_blocks: 2               # 每个分辨率的 ResNet 块的数量
      attn_resolutions: [32]          # 应用注意力机制的位置，即在分辨率等于多少时，应用注意力
      dropout: 0.0                    # Dropout 率
    
    # 损失函数配置
    lossconfig:
      target: movqgan.losses.vqgan_loss.VQGANLoss   # 损失函数类的路径
      params:
        disc_conditional: false       # 非条件判别器
        disc_in_channels: 3           # 判别器的输入通道数
        disc_num_layers: 2            # 判别器的深度
        disc_start: 1                 # 判别器启动时机
        disc_weight: 0.8              # 判别器损失权重
        codebook_weight: 1.0          # codebook 损失权重
        perceptual_weight: 1.0        # LPIPS 损失权重
```

### 损失函数组成

训练目标结合了多个损失：

1. **重建损失（Reconstruction Loss）**：输入与重建之间的像素级 MSE
2. **感知损失（Perceptual Loss）**：使用预训练 VGG 特征的 LPIPS 距离
3. **对抗损失（Adversarial Loss）**：判别器损失（hinge 或 vanilla）
4. **码本损失（Codebook Loss）**：向量量化承诺损失

总损失公式：
```
L = w_rec * L_rec + w_perceptual * L_lpips + w_adv * L_gan + w_codebook * L_vq
```

### 监控训练

训练指标记录到 Weights & Biases：

- `train/aeloss`：自编码器损失
- `train/discloss`：判别器损失
- `train/rec_loss`：重建损失
- `train/p_loss`：感知损失
- `train/g_loss`：生成器对抗损失
- `train/quant_loss`：码本损失
