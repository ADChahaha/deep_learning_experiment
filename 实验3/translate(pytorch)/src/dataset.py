import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def target_operation(encoder_data, decoder_data):
    """
    向量化实现：输入为 (num_samples, seq_len) 的 numpy 数组
    返回：encoder_input, decoder_input, decoder_target
    """
    encoder_input = encoder_data[:, 1:]
    decoder_input = decoder_data[:, :-1]
    decoder_target = decoder_data[:, 1:]
    return encoder_input, decoder_input, decoder_target

def eval_operation(encoder_data, decoder_data):
    encoder_input = encoder_data[:, 1:]
    decoder_input = decoder_data[:, :-1]
    decoder_target = decoder_data[:, 1:]
    return encoder_input, decoder_input, decoder_target

def create_dataset(data_home, batch_size, repeat_num=1, is_training=True,
                   device_num=1, rank=0):
    del repeat_num, device_num, rank

    # 加载数据
    if is_training:
        data_path = os.path.join(data_home, "gru_train.npz")
    else:
        data_path = os.path.join(data_home, "gru_eval.npz")

    with np.load(data_path) as data:
        encoder_data = data["encoder_data"]   # shape: (num_samples, seq_len)
        decoder_data = data["decoder_data"]   # shape: (num_samples, seq_len)

    if is_training:
        enc, dec, tgt = target_operation(encoder_data, decoder_data)
        dataset = TensorDataset(
            torch.from_numpy(enc.astype(np.int64)),
            torch.from_numpy(dec.astype(np.int64)),
            torch.from_numpy(tgt.astype(np.int64))
        )
        collate_fn = lambda batch: {
            "encoder_data": torch.stack([x[0] for x in batch]),
            "decoder_data": torch.stack([x[1] for x in batch]),
            "target_data": torch.stack([x[2] for x in batch]),
        }
    else:
        enc, dec, tgt = eval_operation(encoder_data, decoder_data)
        dataset = TensorDataset(
            torch.from_numpy(enc.astype(np.int64)),
            torch.from_numpy(dec.astype(np.int64)),
            torch.from_numpy(tgt.astype(np.int64))
        )
        collate_fn = lambda batch: {
            "encoder_data": torch.stack([x[0] for x in batch]),
            "decoder_data": torch.stack([x[1] for x in batch]),
            "target_data": torch.stack([x[2] for x in batch]),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        collate_fn=collate_fn
    )

    return dataloader
