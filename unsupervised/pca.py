import numpy as np
import yaml
from dataset import MnistDataset
from utils import visualize


def Z_centered(dataMat: np.ndarray):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)            # 按列求均值
    meanMat = np.tile(meanVal, (rows, 1))         # rows 行的均值矩阵
    newdata = dataMat - meanMat                   # 中心化
    return newdata, meanMat

def EigDV(covMat: np.ndarray, k: int):
    # 协方差矩阵对称，可用 eig；取实部防止数值导致的复数
    D, V = np.linalg.eig(covMat)
    D = np.real(D)
    V = np.real(V)
    idx = np.argsort(D)                           # 从小到大索引
    K_idx = idx[-1:-(k + 1):-1]                   # 取最大的 k 个
    K_vec = V[:, K_idx]                           # 对应特征向量
    return D[K_idx], K_vec

def getlowDataMat(DataMat: np.ndarray, K_eigenVector: np.ndarray):
    return DataMat @ K_eigenVector

def Reconstruction(lowDataMat: np.ndarray, K_eigenVector: np.ndarray, meanVal: np.ndarray):
    return lowDataMat @ K_eigenVector.T + meanVal

def PCA_np(data: np.ndarray, k: int):
    # data: [N, D]
    dataMat = np.asarray(data, dtype=np.float32)   # 兼容 NumPy 2.0（替代 np.mat）
    dataMat, meanVal = Z_centered(dataMat)
    covMat = np.cov(dataMat, rowvar=False)         # 列为特征
    _, V = EigDV(covMat, k)
    lowDataMat = getlowDataMat(dataMat, V)
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return np.asarray(lowDataMat), np.asarray(reconDataMat)


if __name__ == "__main__":
    # 从 config.yaml 读取配置
    with open("config.yaml", "r") as f:
        cfg_yaml = yaml.safe_load(f)
    embed_size = int(cfg_yaml["model"]["embed_channel"])
    embed_size = 30
    path = cfg_yaml["data"]["path"]

    # 读取测试集前 6 张
    dataset = MnistDataset(path, mode="test")
    num_samples = 6
    images = []
    labels = []
    for i in range(num_samples):
        img, label = dataset[i]
        images.append(img.numpy())                 # [1, H, W]
        labels.append(int(label.numpy()))
    images = np.asarray(images)                    # [N, 1, H, W]
    labels = np.asarray(labels)

    N, C, H, W = images.shape
    X = images.reshape(N, -1).astype(np.float32)   # [N, H*W]

    # PCA 降维与重构
    lowData, reconData = PCA_np(X, embed_size)
    reconImages = reconData.reshape(N, C, H, W)

    print("原始图像：")
    visualize(images, labels)
    print(f"PCA 重构图像（压缩到 {embed_size} 维）：")
    visualize(reconImages, labels)

    original_dim = H * W
    compression_ratio = (1 - embed_size / original_dim) * 100
    print(f"\n压缩率: {compression_ratio:.2f}%")
    print(f"原始维度: {original_dim}, 压缩后维度: {embed_size}")
    print(f"数据集路径: {path}")