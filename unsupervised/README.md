# 无监督学习实验（PCA & VAE）

> 实现主成分分析（PCA）与变分自编码器（VAE）在 MNIST 数据集上的无监督学习

## ⬇️ 数据集准备

本实验使用 MNIST 数据集。

**下载命令：**

```bash
wget https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/MNIST.zip
unzip MNIST.zip
rm MNIST.zip
```

解压后将 `MNIST/` 文件夹放入 `data/` 目录下。

实验报告 ipynb 示例路径：

- 数据集：`data/MNIST/`

## 🏗️ 实验内容与目标

- 理解无监督学习方法在图像降维与生成建模中的应用。
- 掌握 PCA 与 VAE 的基本原理与实现。
- 使用 MindSpore 框架实现 PCA 和 VAE。

## 📚 实验原理简介

- **PCA**：线性降维方法，将高维数据投影到低维空间。
- **VAE**：生成式模型，通过编码器-解码器结构学习数据分布。

## 🏗️ 模型结构

- **PCA**: 数据降维
- **VAE**: 编码器（Encoder）→ 潜变量（Latent）→ 解码器（Decoder）

## 🎛️ 超参数配置

修改 `config.yaml` 中的超参数配置：

| 参数          | 说明              |
| ------------- | ----------------- |
| `max_epochs`  | 训练轮数          |
| `lr`          | 学习率            |
| `batch_size`  | 批次大小          |
| `num_workers` | 数据加载线程数    |
| `latent_dim`  | 潜变量维度（VAE） |

## 🚀 快速开始

### 配置环境

请根据依赖手动创建环境并安装 requirements。

### 训练模型

```bash
python train.py
```

## 📁 项目结构

```
.
├── config.yaml
├── datamodule.py
├── dataset.py
├── model.py
├── modelmodule.py
├── pca.py
├── train.py
├── utils.py
├── vae.py
├── assets/
├── checkpoint/
├── data/
│   └── MNIST/
├── lightning_logs/
├── 实验报告.ipynb
└── 无监督学习实验.ipynb
```
