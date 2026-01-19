import torch
import os
import re
from utils import Vocab 


class Multi30K(torch.utils.data.Dataset):
    """Multi30K数据集加载器, 加载Multi30K数据集并处理为一个tensor"""

    def __init__(self, path, de_vocab: Vocab, en_vocab: Vocab):
        self.data = self._load(path)
        # 将所有的单词tokenize,并转换为tensor
        self.data = [
            (torch.tensor((de_vocab.encode(de))), torch.tensor(en_vocab.encode(en)))
            for de, en in self.data
        ]

    def _load(self, path):
        def tokenize(text):
            text = text.rstrip()
            return [tok.lower() for tok in re.findall(r"\w+|[^\w\s]", text)]

        def read_data(data_file_path):
            with open(data_file_path, "r", encoding="utf-8") as data_file:
                data = data_file.readlines()[:-1]
                return [tokenize(i) for i in data]

        members = {i.split(".")[-1]: path + i for i in os.listdir(path)}
        ret = [read_data(members["de"]), read_data(members["en"])]
        return list(zip(*ret))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_dataset = Multi30K("assets/datasets/train/")
    for en, de in train_dataset:
        print(en)
