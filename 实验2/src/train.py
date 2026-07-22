import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .datasets import create_train_loader
from .evaluate import evaluate_model
from .utils import (
    SoftmaxCrossEntropyLoss,
    build_lr_schedule,
    build_model,
    freeze_batchnorm_layers,
    get_device,
    get_ignore_label,
    load_config,
    plot_training_curve,
    resolve_path,
    save_checkpoint,
    segmentation_loss,
    set_seed,
)


class BuildTrainNetwork(nn.Module):
    def __init__(self, network: nn.Module, criterion: nn.Module) -> None:
        super().__init__()
        self.network = network
        self.criterion = criterion

    def forward(self, input_data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        output = self.network(input_data)
        return self.criterion(output, label)


def train_one_run(config: Dict) -> Dict:
    base_dir = Path(__file__).resolve().parents[1]
    set_seed(int(config.get("seed", 42)))
    device = get_device()

    train_loader, voc_root = create_train_loader(config, base_dir)
    print(
        f"Loaded {len(train_loader.dataset)} training images, {len(train_loader)} steps per epoch.",
        flush=True,
    )
    print(f"VOC root: {voc_root}", flush=True)

    network = build_model(config, phase="train").to(device)
    criterion = SoftmaxCrossEntropyLoss(
        num_cls=int(config.get("num_classes", config.get("model", {}).get("num_classes", 21))),
        ignore_label=get_ignore_label(config),
    )
    train_net = BuildTrainNetwork(network, criterion).to(device)

    train_cfg = config.get("train", {})
    paths_cfg = config["paths"]
    epochs = int(train_cfg.get("epochs", config.get("train_epochs", 3)))
    momentum = float(train_cfg.get("momentum", 0.9))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    log_every = int(train_cfg.get("log_every", 20))
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))

    total_steps = len(train_loader) * epochs
    lr_values = build_lr_schedule(config, total_steps)
    optimizer = torch.optim.SGD(
        params=train_net.parameters(),
        lr=lr_values[0] if lr_values else float(train_cfg.get("lr", 1e-3)),
        momentum=momentum,
        weight_decay=weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "miou": [],
        "pixel_accuracy": [],
        "mean_class_accuracy": [],
    }
    log_dir = resolve_path(base_dir, Path(paths_cfg["output_dir"]) / "tensorboard")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    start = time.time()
    global_step = 0
    checkpoint_dir = resolve_path(base_dir, paths_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"best_{paths_cfg['checkpoint_name']}"
    last_checkpoint_path = checkpoint_dir / paths_cfg["checkpoint_name"]
    best_miou = float("-inf")
    best_epoch = 0

    for epoch in range(epochs):
        train_net.train()
        if bool(config.get("freeze_bn", config.get("model", {}).get("freeze_bn", True))):
            freeze_batchnorm_layers(train_net)

        epoch_loss = 0.0
        interval_loss = 0.0
        interval_steps = 0
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Starting epoch {epoch + 1}/{epochs}", flush=True)

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            current_index = min(global_step, len(lr_values) - 1)
            current_lr = lr_values[current_index] if lr_values else optimizer.param_groups[0]["lr"]
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            loss = train_net(images, masks)
            (loss / accumulation_steps).backward()

            if step % accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss += float(loss.item())
            interval_loss += float(loss.item())
            interval_steps += 1

            writer.add_scalar("train/lr", float(current_lr), global_step)
            if step == 1 or step % log_every == 0 or step == len(train_loader):
                avg_interval_loss = interval_loss / max(interval_steps, 1)
                writer.add_scalar("train/loss_interval", avg_interval_loss, global_step)
                print(
                    f"epoch: {epoch + 1}, step: {step}/{len(train_loader)}, "
                    f"avg_loss: {avg_interval_loss:.6f}, lr: {current_lr:.6f}",
                    flush=True,
                )
                interval_loss = 0.0
                interval_steps = 0

        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
        print(
            f"epoch: {epoch + 1}, train_loss: {avg_loss:.6f}, epoch_seconds: {time.time() - epoch_start:.2f}",
            flush=True,
        )

        print(f"Starting evaluation for epoch {epoch + 1}/{epochs}...", flush=True)
        eval_result = evaluate_model(
            config,
            model=network,
            voc_root=voc_root,
            save_visualizations=(epoch + 1 == epochs),
            metrics_filename=f"eval_metrics_epoch_{epoch + 1}.json",
        )
        epoch_miou = float(eval_result["miou"])
        epoch_pixel_accuracy = float(eval_result["pixel_accuracy"])
        epoch_mean_class_accuracy = float(eval_result["mean_class_accuracy"])
        history["miou"].append(epoch_miou)
        history["pixel_accuracy"].append(epoch_pixel_accuracy)
        history["mean_class_accuracy"].append(epoch_mean_class_accuracy)
        writer.add_scalar("eval/miou", epoch_miou, epoch + 1)
        writer.add_scalar("eval/pixel_accuracy", epoch_pixel_accuracy, epoch + 1)
        writer.add_scalar("eval/mean_class_accuracy", epoch_mean_class_accuracy, epoch + 1)
        print(
            f"epoch: {epoch + 1}, val_miou: {epoch_miou:.6f}, "
            f"pixel_accuracy: {epoch_pixel_accuracy:.6f}, "
            f"mean_class_accuracy: {epoch_mean_class_accuracy:.6f}",
            flush=True,
        )

        save_checkpoint(network, config, last_checkpoint_path, history)
        if epoch_miou > best_miou:
            best_miou = epoch_miou
            best_epoch = epoch + 1
            save_checkpoint(network, config, best_checkpoint_path, history)
            print(
                f"Best checkpoint updated at epoch {best_epoch}: mIoU={best_miou:.6f}",
                flush=True,
            )

    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.2f}s", flush=True)
    print(
        f"Best epoch: {best_epoch}, best_mIoU: {best_miou:.6f}, "
        f"best_checkpoint: {best_checkpoint_path}",
        flush=True,
    )

    output_dir = resolve_path(base_dir, paths_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "train_loss_history": history["train_loss"],
        "miou_history": history["miou"],
        "pixel_accuracy_history": history["pixel_accuracy"],
        "mean_class_accuracy_history": history["mean_class_accuracy"],
        "final_train_loss": history["train_loss"][-1],
        "final_miou": history["miou"][-1],
        "final_pixel_accuracy": history["pixel_accuracy"][-1],
        "final_mean_class_accuracy": history["mean_class_accuracy"][-1],
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path),
        "checkpoint_path": str(last_checkpoint_path),
        "voc_root": str(voc_root),
        "device": str(device),
        "train_seconds": elapsed,
        "num_train_samples": len(train_loader.dataset),
        "steps_per_epoch": len(train_loader),
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_training_curve(history["train_loss"], output_dir / "train_loss_curve.png")
    writer.flush()
    writer.close()
    return metrics


def main() -> None:
    config = load_config("config.json")
    train_one_run(config)


if __name__ == "__main__":
    main()
