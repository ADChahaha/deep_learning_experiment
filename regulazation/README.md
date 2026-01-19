# 神经网络正则化实验

> 探索 BatchNorm、Dropout、L2 正则等方法对神经网络的影响

## ⬇️ 数据集准备

本实验使用 Fashion-MNIST 数据集。

**下载命令：**

```bash
wget https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/fashion-mnist.zip
unzip fashion-mnist.zip
rm fashion-mnist.zip
```

解压后将 `fashion-mnist/` 文件夹放入 `data/` 目录下。

实验报告 ipynb 示例路径：

- 训练集：`data/fashion-mnist/train/`
- 测试集：`data/fashion-mnist/test/`

## 🏗️ 实验内容与目标

- 理解正则化（BatchNorm、Dropout、L2、Early Stop）对神经网络过拟合和优化的作用及特点。
- 在 MindSpore 框架下实现并对比不同正则化方法的效果。

## 📚 实验原理简介

- **Batch Normalization**：对神经网络中间层进行归一化，提升训练稳定性和速度。
- **Dropout**：训练时随机丢弃部分神经元，防止过拟合。
- **L2 正则**：在损失函数中加入参数范数约束，抑制模型复杂度。
- **Early Stop**：通过验证集监控，防止模型在训练集上过拟合。

## 🏗️ 模型结构

```
输入层 → 多层全连接/卷积层（可选正则化）→ 输出层
```

## 🎛️ 超参数配置

修改 `config.yaml` 中的超参数配置：

| 参数          | 说明           |
| ------------- | -------------- |
| `max_epochs`  | 训练轮数       |
| `lr`          | 学习率         |
| `batch_size`  | 批次大小       |
| `num_workers` | 数据加载线程数 |
| `reg_type`    | 正则化类型     |
| `reg_lambda`  | L2 正则系数    |

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
├── result.ipynb
├── 正则化实验.ipynb
├── train.py
├── utils.py
├── asset/
├── data/
│   └── fashion-mnist/
├── lightning_logs/
└── 实验报告.docx
```
