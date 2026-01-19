# 基于 MLP 的汽车里程数预测

> 使用多层感知机 (MLP) 预测汽车燃油效率 (MPG)

## ⬇️ 数据集下载

```python
from download import download

# 下载汽车里程 auto-mpg 数据集
url = "https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/deep-learning/auto-mpg.zip"
path = download(url, "./", kind="zip", replace=True)
```

> ⚠️ **注意**: 请创建`data/`文件夹，并将下载后的文件解压放入
> **汽车里程数回归预测实验-函数式自动微分修改版.ipynb 中数据集路径举例：**
>
> - `./auto-mpg.data`（即项目根目录或 `data/auto-mpg.data`，请确保与代码一致）
> - 代码如 `with open('./auto-mpg.data') as csv_file:`，注意路径相对性。

## 🏗️ 模型架构

```
输入层 (9) → 全连接层 (64) → ReLU → 全连接层 (64) → ReLU → 输出层 (1)
```

```python
self.fc1 = nn.Linear(9, 64)
self.relu1 = nn.ReLU()
self.fc2 = nn.Linear(64, 64)
self.relu2 = nn.ReLU()
self.fc3 = nn.Linear(64, 1)
```

## 🎛️ 超参数配置

修改 `config.yaml` 中的超参数配置：

| 参数                      | 说明                   |
| ------------------------- | ---------------------- |
| `max_epochs`              | 训练轮数               |
| `check_val_every_n_epoch` | 验证间隔轮数           |
| `save_top_k`              | 保存最佳模型数量       |
| `lr`                      | 学习率                 |
| `batch_size`              | 批次大小               |
| `num_workers`             | 数据加载线程数         |
| `path_data`               | 数据集路径             |
| `ratio`                   | 训练集与验证集划分比例 |

> 本实验使用MacOS系统进行，Linux系统请根据实际情况调整`num_workers`参数。

## 🚀 快速开始

### 配置环境

```bash
conda env create -f environment.yaml
```

### 训练模型

```bash
python train.py fit -c config.yaml
```

## 📁 项目结构

```
.
├── config.yaml
├── data
│   ├── auto-mpg.data
├── datamodule.py
├── dataset.py
├── environment.yml
├── model.py
├── modelmodule.py
├── README_CN.md
├── README.md
├── train.py
└── 汽车里程数回归预测实验-函数式自动微分修改版.ipynb
```
