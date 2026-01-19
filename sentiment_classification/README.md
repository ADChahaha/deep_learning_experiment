# 基于 RNN 的情感分类实验

> 使用循环神经网络 (RNN) 对文本进行情感分类

## ⬇️ 数据集准备

本实验使用 IMDB 影评数据集和 Glove 词向量。

**IMDB 数据集下载：**
[https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)

**Glove 词向量下载：**
[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

下载后将解压内容分别放入 `data/` 目录下。

实验报告 ipynb 示例路径：

- 数据集：`data/`
- 词向量文件：`data/glove.6B.100d.txt` 等

## 🏗️ 实验内容与目标

- 理解循环神经网络（RNN）在文本情感分类中的应用。
- 掌握词嵌入、序列建模与分类流程。
- 使用 MindSpore 框架实现文本情感分类。

## 📚 实验原理简介

- **Embedding**：将词语映射为稠密向量。
- **RNN/LSTM/GRU**：建模文本序列特征。
- **Linear + Softmax**：输出情感类别概率。

## 🏗️ 模型结构

```
Embedding → RNN/LSTM/GRU → Linear → Softmax
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
| `path_glove`  | 词向量路径     |

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
├── datamodule.py
├── dataset.py
├── eval.py
├── model.py
├── modelmodule.py
├── result.ipynb
├── 情感分类实验.ipynb
├── train.py
├── utils.py
├── asset/
├── checkpoint/
├── data/
├── lightning_logs/
└── 实验报告.docx
```
