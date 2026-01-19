import torch
import numpy as np
from modelmodule import MyModel
from dataset import MnistDataset
from utils import visualize


def reconstruct_images(model, dataset, num_samples=6):
    """用训练好的 AutoEncoder 重构图像"""
    model.eval()
    images = []
    labels = []
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(num_samples):
            img, label = dataset[i]
            img = img.unsqueeze(0)  # [1, 1, H, W]
            img = img.to(device)  # ← 移到模型设备
            
            x, lowData, reconData = model.model(img)
            
            images.append(img.cpu().squeeze(0).numpy())  # ← 移回 CPU 后 numpy
            labels.append(int(label.numpy()))
            
            if i == 0:
                recon_batch = reconData
            else:
                recon_batch = torch.cat([recon_batch, reconData], dim=0)
    
    # 恢复重构图像形状
    recon_images = recon_batch.cpu().view(num_samples, 1, 28, 28).numpy()  # ← 移回 CPU
    return np.asarray(images), np.asarray(recon_images), np.asarray(labels)


if __name__ == "__main__":
    embed_channel=2
    # 加载训练好的模型
    model = MyModel.load_from_checkpoint(
        f"checkpoint/dim{embed_channel}.ckpt",
        in_channel=784,
        embed_channel=embed_channel
    )
    
    # 加载测试数据集
    dataset = MnistDataset("data/MNIST", mode="test")
    
    # 重构图像
    ori_images, recon_images, labels = reconstruct_images(model, dataset, num_samples=6)
    
    # 可视化对比
    print("原始图像：")
    visualize(ori_images, labels)
    
    print(f"AutoEncoder 重构图像（压缩到 {embed_channel} 维）：")
    visualize(recon_images, labels)