class Vocab:
    """通过词频字典，构建词典"""

    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, word_count_dict, min_freq=1):
        self.word2idx = {}
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx

        filted_dict = {w: c for w, c in word_count_dict.items() if c >= min_freq}
        for w, _ in filted_dict.items():
            self.word2idx[w] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.bos_idx = self.word2idx['<bos>']
        self.eos_idx = self.word2idx['<eos>']
        self.pad_idx = self.word2idx['<pad>']
        self.unk_idx = self.word2idx['<unk>']

    def _word2idx(self, word):
        """单词映射至数字索引"""
        if word not in self.word2idx:
            return self.unk_idx
        return self.word2idx[word]

    def _idx2word(self, idx):
        """数字索引映射至单词"""
        if idx not in self.idx2word:
            raise ValueError(f'input index: {idx} is not in vocabulary.')
        return self.idx2word[idx]

    def encode(self, word_or_list):
        """将单个单词或单词数组映射至单个数字索引或数字索引数组"""
        if isinstance(word_or_list, list):
            return [self._word2idx(i) for i in word_or_list]
        return self._word2idx(word_or_list)

    def decode(self, idx_or_list):
        """将单个数字索引或数字索引数组映射至单个单词或单词数组"""
        if isinstance(idx_or_list, list):
            return [self._idx2word(i) for i in idx_or_list]
        return self._idx2word(idx_or_list)

    def __len__(self):
        return len(self.word2idx)

from collections import Counter, OrderedDict

def build_vocab(dataset):
    de_words, en_words = [], []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)
    
    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))

    return Vocab(de_count_dict, min_freq=2), Vocab(en_count_dict, min_freq=2)

import json
class LoadVocab(Vocab):

    def __init__(self, word2idx_path:str, idx2word_path:str):
        if word2idx_path is not None and word2idx_path != "":
            with open(word2idx_path, 'r', encoding="utf-8") as f:
                self.word2idx = json.load(f)
        if idx2word_path is not None and idx2word_path != "":
            with open(idx2word_path, 'r', encoding="utf-8") as f:
                self.idx2word = json.load(f)

        self.bos_idx = self.word2idx['<bos>']
        self.eos_idx = self.word2idx['<eos>']
        self.pad_idx = self.word2idx['<pad>']
        self.unk_idx = self.word2idx['<unk>']


