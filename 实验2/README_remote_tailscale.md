# 实验2 远程训练说明

这个目录已经包含训练所需的代码、VOC2012 数据和预训练权重，可以通过 Tailscale 把整个实验临时同步到另一台机器训练，训练结束后再把日志和 checkpoint 拉回本机。

## 远端机器需要准备

- 能通过 Tailscale SSH 或普通 SSH 访问
- 已安装 Python 3
- 已准备好 `torch`、`torchvision`、`numpy`、`matplotlib`、`Pillow`、`tensorboard`

可参考依赖文件：

```bash
pip install -r requirements-remote.txt
```

如果远端使用 conda 或虚拟环境，可以通过环境变量 `REMOTE_ACTIVATE` 在训练前激活。

## 一条命令运行

在本机的 `实验2` 目录执行：

```bash
chmod +x scripts/run_remote_tailscale.sh
./scripts/run_remote_tailscale.sh <tailscale-host> [remote-python]
```

示例：

```bash
./scripts/run_remote_tailscale.sh gpu-box.tailnet.ts.net
./scripts/run_remote_tailscale.sh 100.101.102.103 /opt/venvs/dl/bin/python
```

如果远端需要先激活 conda：

```bash
REMOTE_ACTIVATE='source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch' \
./scripts/run_remote_tailscale.sh gpu-box.tailnet.ts.net
```

## Asus 这台 Windows + Git Bash 机器

这台机器已经确认可用信息如下：

- SSH: `asus@asus.tailc3e9be.ts.net`
- Git Bash: `C:\Program Files\Git\bin\bash.exe`
- 可用训练环境: `segexp`
- 训练 Python:

```bash
/c/Users/14195/miniconda3/envs/segexp/python.exe
```

直接运行：

```bash
chmod +x scripts/run_remote_tailscale_asus_gitbash.sh
./scripts/run_remote_tailscale_asus_gitbash.sh
```

这个脚本会通过 Git Bash 在远端创建临时目录、传输训练文件、执行训练、回传 `outputs/` 和 `checkpoints/`，然后删除远端临时目录。

## 脚本会做什么

1. 在远端 `/tmp` 下创建临时目录
2. 同步 `src`、`config.json`、`data`、`checkpoints` 等训练所需文件
3. 在远端执行 `python -m src.train`
4. 把 `outputs/`、`checkpoints/`、训练日志拉回本地 `outputs/remote_runs/<时间戳>/`
5. 删除远端临时目录

## 回传内容

- `outputs/remote_train.log`
- `outputs/train_metrics.json`
- `outputs/eval_metrics.json`
- `outputs/tensorboard/`
- `outputs/visualizations/`
- `checkpoints/*.pth`

## 注意

- 当前脚本默认远端已有完整依赖，不会自动安装 PyTorch。
- 当前会同步 `data/`，如果远端机器和本机共享同一份数据目录，可以后续再改成只同步代码。
- 我本机上执行 `tailscale status --json` 时 CLI 返回了 `Failed to load preferences`，所以脚本目前按你提供的主机名或 Tailscale IP 来连接，不自动枚举节点。
- `asus` 这台机器没有远端 `rsync`，所以 Windows 版脚本使用的是 `tar + ssh` 流式传输。
