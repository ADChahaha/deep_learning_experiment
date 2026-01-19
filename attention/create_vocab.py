from utils import Vocab, build_vocab
import re
import os

def load(path):
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

if __name__ == "__main__":
    dataset = load("assets/datasets/train/")
    de_vocab, en_vocab = build_vocab(dataset)

    # 写入文件
    import json
    with open("assets/vocabs/de_vocab.json", 'w', encoding="utf-8") as f:
        json.dump(de_vocab.word2idx, f)
    with open("assets/vocabs/en_vocab.json", 'w', encoding="utf-8") as f:
        json.dump(en_vocab.word2idx, f)
    with open("assets/vocabs/de_idx_vocab.json", 'w', encoding="utf-8") as f:
        json.dump(de_vocab.idx2word, f)
    with open("assets/vocabs/en_idx_vocab.json", 'w', encoding="utf-8") as f:
        json.dump(en_vocab.idx2word, f)

