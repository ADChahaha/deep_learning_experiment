import numpy as np
import sys
import torch


def load_glove(path: str):
    embeddings = []
    tokens = []
    # 读取文件
    try:
        with open(path, "r", encoding="utf-8") as gf:
            for glove in gf:
                word, embedding = glove.split(maxsplit=1)
                tokens.append(word)
                embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=" "))
    except FileNotFoundError as e:
        print(f"读取词嵌入文件时出错，路径不存在")
        sys.exit(1)

    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding和tokens
    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))
    tokens.append("<unk>")
    tokens.append("<pad>")

    vocab = {tok: idx for idx, tok in enumerate(tokens)}
    embedding_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32)
    return vocab, embedding_tensor


