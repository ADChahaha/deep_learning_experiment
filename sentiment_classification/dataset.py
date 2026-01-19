import torch
import re
import string
import tarfile


class MovieDataset(torch.utils.data.Dataset):
    """IMDB数据集加载器

    加载IMDB数据集并处理为一个Python迭代对象。

    """

    label_map = {"pos": 1, "neg": 0}

    def __init__(self, path: str, vocab: dict, mode="train") -> None:
        self.mode = mode
        self.path = path
        self.docs, self.labels = [], []
        self.vocab = vocab
        self.unk_idx = self.vocab["<unk>"]

        # 将数据load到 self.docs 和 self.labels 两个数组里
        self._load("pos")
        self._load("neg")

        # 将整个数组的str转换为id
        self.docs = self._token2id(self.docs)

    def _load(self, label: str) -> None:
        pattern = re.compile(r"aclImdb/{}/{}/.*\.txt$".format(self.mode, label))
        with tarfile.open(self.path) as tarf:
            tf = tarf.next()
            while tf is not None:
                if bool(pattern.match(tf.name)):
                    # 读取并解码为字符串
                    text = tarf.extractfile(tf).read().decode("utf-8")
                    # 去除换行符
                    text = text.strip()
                    # 去除标点符号
                    text = text.translate(str.maketrans("", "", string.punctuation))
                    # 转小写并分词
                    tokens = text.lower().split()

                    self.docs.append(tokens)
                    self.labels.append([self.label_map[label]])
                tf = tarf.next()

    def _token2id(self, docs: list[list[str]]) -> list[list[int]]:
        return [[self.vocab.get(word, self.unk_idx) for word in seq] for seq in docs]

    def __getitem__(self, idx):
        return torch.tensor(self.docs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.docs)


if __name__ == "__main__":
    train_dataset = MovieDataset(
        "data/aclImdb_v1.tar.gz", {"<unk>": 400000}, mode="train"
    )
    print(len(train_dataset))
    print(train_dataset[1])
