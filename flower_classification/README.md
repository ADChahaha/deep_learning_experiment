# 基于卷积神经网络的花卉识别

> 使用卷积神经网络 (CNN) 对花卉图片进行分类识别

## ⬇️ 数据集准备

请下载 [flower_photos 数据集](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)，并解压到 `data/flower_photos/` 目录下。

> **花卉识别.ipynb 中数据集路径举例：**
>
> - `cfg['data_path'] = 'flower_photos'`，即数据集主目录为 `data/flower_photos/`，代码中如 `glob.glob(f'flower_photos/{titles[i]}/*.jpg')`。

> ⚠️ **注意**: 数据集目录结构应如下：
>
> ```
> data/flower_photos/
> ├── daisy/
> ├── dandelion/
> ├── roses/
> ├── sunflowers/
> └── tulips/
> ```

## 🏗️ 模型架构

默认模型结构如下：

```
输入层 (3x224x224) → 卷积层/池化层若干 → 全连接层 → Softmax 输出 (5 类)
```

示例（OriginNet）：

```python
self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
self.relu1 = nn.ReLU()
self.pool1 = nn.MaxPool2d(2, 2)
self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
self.relu2 = nn.ReLU()
self.pool2 = nn.MaxPool2d(2, 2)
self.fc1 = nn.Linear(64 * 56 * 56, 128)
self.relu3 = nn.ReLU()
self.fc2 = nn.Linear(128, 5)
```

## 🎛️ 超参数配置

修改 `config.yaml` 中的超参数配置：

| 参数                      | 说明             |
| ------------------------- | ---------------- |
| `max_epochs`              | 训练轮数         |
| `check_val_every_n_epoch` | 验证间隔轮数     |
| `save_top_k`              | 保存最佳模型数量 |
| `lr`                      | 学习率           |
| `batch_size`              | 批次大小         |
| `num_workers`             | 数据加载线程数   |
| `path_data`               | 数据集路径       |
| `img_size`                | 输入图片尺寸     |
| `model_name`              | 使用的模型结构   |

> MacOS 用户建议将 `num_workers` 设置为 0 或 2，Linux 可适当调高。

## 🚀 快速开始

### 配置环境

```bash
conda env create -f environment.yml
```

### 训练模型

```bash
python train.py fit -c config.yaml
```

## 📁 项目结构

```
.
├── config.yaml
├── datamodule.py
├── dataset.py
├── environment.yml
├── model.py
├── modelmodule.py
├── README_CN.md
├── README.md
├── train.py
├── result.ipynb
├── asset/
│   ├── OriginNet/
│   └── OriginNetWithRes/
├── data/
│   └── flower_photos/
└── lightning_logs/
    ├── OriginNet/
    └── OriginNetWithRes/
```
