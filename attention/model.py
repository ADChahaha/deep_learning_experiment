import torch.nn as nn
from utils import LoadVocab
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # batch 维度
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class Model(nn.Module):
    def __init__(
        self,
        de_vocab: LoadVocab,
        en_vocab: LoadVocab,
        embedding_dim: int,
        nhead: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
    ) -> None:
        super().__init__()
        # 词向量嵌入
        self.de_embedding = nn.Embedding(len(de_vocab), embedding_dim, de_vocab.pad_idx)
        self.en_embedding = nn.Embedding(len(en_vocab), embedding_dim, en_vocab.pad_idx)
        # 位置编码
        self.position_encoding = PositionalEncoding(embedding_dim)
        # transformer Layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, batch_first=True),
            num_layers=encoder_num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, nhead, batch_first=True),
            num_layers=decoder_num_layers,
        )
        self.fc = nn.Linear(embedding_dim, len(en_vocab))
        self.en_bos_idx = en_vocab.bos_idx
        self.en_eos_idx = en_vocab.eos_idx

    def _generate_tgt_mask(self, tgt_seq_len, device=None):
        """
        返回 tgt_mask, 用于 TransformerDecoder
        tgt_seq_len: 当前 decoder 输入序列长度
        device: mask 所在设备
        输出形状: (tgt_seq_len, tgt_seq_len)
        """
        mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1
        ).bool()  # bool mask
        if device is not None:
            mask = mask.to(device)
        return mask

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        tgt_mask=None,
    ):
        """
        Params:
        src 为 (batch_size, src_seq_len) 的 token id
        tgt 为 (batch_size, tgt_seq_len) 的 token id
        src_key_padding_mask 为一个 (batch_size, max_scc_seq_len) 的 bool tensor, 表示 src 的 padding mask
        tgt_key_padding_mask 为一个 (batch_size, max_tgt_seq_len) 的 bool tensor, 表示 tgt 的 paddig mask
        tgt_mask 为一个二维 tensor, 表示 tgt 的未来 mask
        输出(batch_size, tgt_seq_len, vocab_size) 的概率表, 表示每个 tgt token 的下一词的概率分布
        """
        # 生成输出mask
        if tgt_mask is None:
            tgt_mask = self._generate_tgt_mask(tgt.shape[1], device=tgt.device)
        # 输入embedding
        src_embedded = self.de_embedding(src)
        src_embedded = self.position_encoding(src_embedded)
        # 输出embedding
        tgt_embedded = self.en_embedding(tgt)
        tgt_embedded = self.position_encoding(tgt_embedded)
        # encoder
        memory = self.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)
        # decoder
        output = self.decoder(
            tgt_embedded,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc(output)
        return logits

    @torch.no_grad()
    def generate(self, src, src_key_padding_mask, max_seq_len: int = 32):
        """
        假设 src 是一个 (batch_size, seq)
        返回一个翻译后的句子
        """
        device = src.device
        B = src.size(0)

        # 生成 encoder 之后的 memory
        src_embedded = self.de_embedding(src)
        src_embedded = self.position_encoding(src_embedded)
        memory = self.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)

        # decoder 输入初始化
        tgt = torch.full((B, 1), self.en_bos_idx, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # 自回归生成
        # 每次都保存最原始的 tgt. 即 ids
        for _ in range(max_seq_len):
            tgt_mask = self._generate_tgt_mask(tgt.size(1), device=device)

            tgt_embedded = self.en_embedding(tgt)
            tgt_embedded = self.position_encoding(tgt_embedded)

            output = self.decoder(
                tgt_embedded,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.fc(output)
            # 取最后一个 logits
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=1)

            next_token = torch.where(
                finished, torch.full_like(next_token, self.en_eos_idx), next_token
            )
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            # 更新finished
            finished |= (next_token == self.en_eos_idx)

            if finished.all():
                break
        return tgt


if __name__ == "__main__":
    # 加载模型
    model = Model(
        LoadVocab("assets/vocabs/de_vocab.json", "assets/vocabs/de_idx_vocab.json"),
        LoadVocab("assets/vocabs/en_vocab.json", "assets/vocabs/en_idx_vocab.json"),
        128,
        8,
        4,
        4,
    )
    # 参数配置
    batch_size = 16
    input_seq_len = 28
    output_seq_len = 15
    # 生成随机输入数据
    src = torch.randint(0, 100, (batch_size, input_seq_len))
    tgt = torch.randint(0, 100, (batch_size, output_seq_len))
    src_lengths = torch.randint(input_seq_len - 10, input_seq_len, (batch_size,))
    tgt_lengths = torch.randint(output_seq_len - 10, output_seq_len, (batch_size,))
    src_idx = (
        torch.arange(0, input_seq_len).unsqueeze(0).expand(batch_size, input_seq_len)
    )
    src_key_padding_mask = src_idx > src_lengths.unsqueeze(1)
    tgt_idx = (
        torch.arange(0, output_seq_len).unsqueeze(0).expand(batch_size, output_seq_len)
    )
    tgt_key_padding_mask = tgt_idx > tgt_lengths.unsqueeze(1)
    # 查看模型输出是否正确
    logits = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
