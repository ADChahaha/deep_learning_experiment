# 基于注意力机制的神经机器翻译实验

> 使用注意力机制实现德语到英语的神经机器翻译（NMT）

## ⬇️ 数据集准备

本实验使用 Multi30K 数据集。

**下载命令：**

```python
pip install download
from download import download
url = "https://modelscope.cn/api/v1/datasets/SelinaRR/Multi30K/repo?Revision=master&FilePath=Multi30K.zip"
download(url, './', kind='zip', replace=True)
```

解压后将 `datasets/` 目录下的 `train/`、`valid/`、`test/` 放入 `assets/datasets/`。

或手动下载并解压到 `assets/datasets/`。

实验报告 ipynb 示例路径：

- 训练集：`assets/datasets/train/`
- 词表文件：`assets/vocabs/`

## 🏗️ 实验内容与目标

- 理解注意力机制在神经机器翻译中的作用。
- 掌握编码器-解码器结构与注意力机制的结合。
- 使用 MindSpore 框架实现带注意力的 NMT。

## 📚 实验原理简介

- **编码器-解码器**：将源语言编码为上下文向量，再解码为目标语言。
- **注意力机制**：为每个输出动态分配输入的关注权重，提升长序列翻译效果。

## 🏗️ 模型结构

```
编码器（Embedding → RNN）→ 注意力机制 → 解码器（RNN → Linear）
```

## 🎛️ 超参数配置

修改 `config.yaml` 中的超参数配置：

| 参数          | 说明           |
| ------------- | -------------- |
| `max_epochs`  | 训练轮数       |
| `lr`          | 学习率         |
| `batch_size`  | 批次大小       |
| `num_workers` | 数据加载线程数 |
| `hidden_size` | 隐藏层维度     |
| `embed_size`  | 词嵌入维度     |
| `path_data`   | 数据集路径     |
| `path_vocab`  | 词表路径       |

## 🚀 快速开始

### 配置环境

请根据依赖手动创建环境并安装 requirements。

### 训练模型

```bash
python train.py
```

### 评估模型

```bash
python eval.py
```

## 📁 项目结构

```
.
├── config.yaml
├── create_vocab.py
├── datamodule.py
├── dataset.py
├── eval.py
├── model.py
├── modelmodule.py
├── train.py
├── utils.py
├── assets/
│   ├── datasets/
│   ├── images/
│   └── vocabs/
├── checkpoint/
├── lightning_logs/
├── 实验报告.ipynb
└── 基于Transformer实现德译英翻译实验.ipynb
```
