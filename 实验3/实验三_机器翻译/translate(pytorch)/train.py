"""Train GRU translation model with PyTorch."""

import argparse
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from src.checkpointing import append_json_record, save_epoch_checkpoint, save_model
from src.config import cfg
from src.dataset import create_dataset
from src.metrics import corpus_bleu, decode_chinese, load_vocab
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


def evaluate_model(model, dataloader, device, ch_vocab):
    infer_model = Seq2Seq(cfg, is_train=False).to(device)
    infer_model.load_state_dict(model.state_dict())
    infer_net = InferCell(infer_model, cfg).to(device)
    infer_net.eval()
    loss_model = WithLossCell(Seq2Seq(cfg, is_train=True).to(device), cfg).to(device)
    loss_model._backbone.load_state_dict(model.state_dict())
    loss_model.eval()

    references = []
    hypotheses = []
    loss_total = 0.0
    batch_count = 0

    with torch.no_grad():
        for data in dataloader:
            src = data["encoder_data"].to(device)
            dst = data["decoder_data"].to(device)
            label = data["target_data"].to(device)
            loss_total += loss_model(src, dst, label).item()
            batch_count += 1
            predictions = infer_net(src, dst)

            for target_ids, predicted_ids in zip(data["target_data"].tolist(), predictions.cpu().tolist()):
                reference_text = decode_chinese(target_ids, ch_vocab)
                hypothesis_text = decode_chinese(predicted_ids, ch_vocab)
                references.append(list(reference_text))
                hypotheses.append(list(hypothesis_text))

    bleu_result = corpus_bleu(references, hypotheses)
    return {
        "bleu": bleu_result["bleu"],
        "eval_loss": loss_total / max(batch_count, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch GRU Example")
    parser.add_argument("--dataset_path", type=str, default="./preprocess", help="dataset path.")
    parser.add_argument("--ckpt_save_path", type=str, default="./ckpt", help="checkpoint save path.")
    parser.add_argument("--log_dir", type=str, default=cfg.log_dir, help="tensorboard log path.")
    parser.add_argument("--history_path", type=str, default=cfg.history_path, help="train history path.")
    parser.add_argument("--num_epochs", type=int, default=cfg.num_epochs, help="number of epochs.")
    args = parser.parse_args()

    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(args.ckpt_save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _ensure_dataset(args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = create_dataset(args.dataset_path, cfg.batch_size)
    ds_eval = create_dataset(args.dataset_path, cfg.eval_batch_size, is_training=False)
    ch_vocab = load_vocab(os.path.join(args.dataset_path, "ch_vocab.txt"))

    backbone = Seq2Seq(cfg).to(device)
    network = WithLossCell(backbone, cfg).to(device) #记录单个批尺寸数据集的损失值
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.98)) #使用Adam优化器
    writer = SummaryWriter(log_dir=args.log_dir)
    best_bleu = float("-inf")

    for epoch in range(1, args.num_epochs + 1):
        network.train()
        epoch_start = time.time()
        epoch_loss_total = 0.0
        step_count = 0

        for step, data in enumerate(ds_train, start=1):
            src = data["encoder_data"].to(device)
            dst = data["decoder_data"].to(device)
            label = data["target_data"].to(device)

            optimizer.zero_grad()
            loss = network(src, dst, label)
            loss.backward()
            optimizer.step()
            epoch_loss_total += loss.item()
            step_count += 1

        epoch_time_ms = (time.time() - epoch_start) * 1000.0
        per_step_time_ms = epoch_time_ms / max(len(ds_train), 1)
        avg_train_loss = epoch_loss_total / max(step_count, 1)
        bleu_result = evaluate_model(backbone, ds_eval, device, ch_vocab)

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("eval/loss", bleu_result["eval_loss"], epoch)
        writer.add_scalar("eval/bleu", bleu_result["bleu"], epoch)
        writer.add_scalar("runtime/epoch_time_ms", epoch_time_ms, epoch)
        writer.flush()

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "eval_loss": bleu_result["eval_loss"],
            "bleu": bleu_result["bleu"],
            "epoch_time_ms": epoch_time_ms,
            "per_step_time_ms": per_step_time_ms,
        }
        append_json_record(record, args.history_path)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "eval_loss": bleu_result["eval_loss"],
            "bleu": bleu_result["bleu"],
            "config": dict(cfg),
        }
        checkpoint_path = save_epoch_checkpoint(args.ckpt_save_path, epoch, checkpoint_payload)

        if bleu_result["bleu"] >= best_bleu:
            best_bleu = bleu_result["bleu"]
            save_model(checkpoint_payload, os.path.join(args.ckpt_save_path, "best_bleu.pt"))

        print(
            f"epoch: {epoch}, train_loss: {avg_train_loss:.7f}, eval_loss: {bleu_result['eval_loss']:.7f}, "
            f"bleu: {bleu_result['bleu']:.2f}, checkpoint: {checkpoint_path}"
        )
        print(f"epoch time: {epoch_time_ms:.3f} ms, per step time: {per_step_time_ms:.3f} ms")

    writer.close()
