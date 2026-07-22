"""Evaluate GRU translation model with PyTorch."""

import argparse
import os
import statistics

import torch

from src.config import cfg
from src.dataset import create_dataset
from src.metrics import corpus_bleu, decode_chinese, decode_english, load_vocab
from src.preprocess import convert_to_mindrecord
from src.seq2seq import InferCell, Seq2Seq, WithLossCell


def _ensure_dataset(dataset_path):
    train_file = os.path.join(dataset_path, "gru_train.npz")
    eval_file = os.path.join(dataset_path, "gru_eval.npz")
    en_vocab = os.path.join(dataset_path, "en_vocab.txt")
    ch_vocab = os.path.join(dataset_path, "ch_vocab.txt")
    if all(os.path.exists(p) for p in [train_file, eval_file, en_vocab, ch_vocab]):
        return
    convert_to_mindrecord("src/cmn_zhsim.txt", dataset_path, cfg.max_seq_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch GRU Example")
    parser.add_argument("--dataset_path", type=str, default="./preprocess", help="dataset path.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path.")
    parser.add_argument("--max_samples", type=int, default=20, help="max printed samples.")
    args = parser.parse_args()

    os.makedirs(args.dataset_path, exist_ok=True)
    _ensure_dataset(args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_eval = create_dataset(args.dataset_path, cfg.eval_batch_size, is_training=False)

    network = Seq2Seq(cfg, is_train=False)
    network = InferCell(network, cfg).to(device)
    network.eval()
    loss_model = WithLossCell(Seq2Seq(cfg, is_train=True).to(device), cfg).to(device)

    checkpoint_path = args.checkpoint_path or cfg.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    network.network.load_state_dict(state_dict)
    loss_model._backbone.load_state_dict(state_dict)
    loss_model.eval()

    en_vocab = load_vocab(os.path.join(args.dataset_path, "en_vocab.txt"))
    ch_vocab = load_vocab(os.path.join(args.dataset_path, "ch_vocab.txt"))

    bleu_references = []
    bleu_hypotheses = []
    sample_matches = []
    loss_total = 0.0
    batch_count = 0
    printed = 0
    with torch.no_grad():
        for data in ds_eval:
            src = data["encoder_data"].to(device)
            dst = data["decoder_data"].to(device)
            label = data["target_data"].to(device)
            loss_total += loss_model(src, dst, label).item()
            batch_count += 1

            output = network(src, dst)

            for encoder_ids, decoder_ids, predicted_ids in zip(
                data["encoder_data"].tolist(),
                data["target_data"].tolist(),
                output.cpu().tolist(),
            ):
                en_data = decode_english(encoder_ids, en_vocab)
                ch_data = decode_chinese(decoder_ids, ch_vocab)
                out = decode_chinese(predicted_ids, ch_vocab)

                bleu_references.append(list(ch_data))
                bleu_hypotheses.append(list(out))
                sample_matches.append(float(ch_data == out))

                if printed < args.max_samples:
                    print("English:", en_data)
                    print("expect Chinese:", ch_data)
                    print("predict Chinese:", out)
                    print(" ")
                    printed += 1

    bleu_result = corpus_bleu(bleu_references, bleu_hypotheses)
    exact_match = 100.0 * statistics.mean(sample_matches) if sample_matches else 0.0
    eval_loss = loss_total / max(batch_count, 1)
    print(f"Eval loss: {eval_loss:.7f}")
    print(f"BLEU: {bleu_result['bleu']:.2f}")
    print(f"Exact match: {exact_match:.2f}%")
