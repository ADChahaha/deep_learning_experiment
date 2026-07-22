import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .datasets import get_eval_ids, load_eval_sample
from .utils import (
    VOC_CLASSES,
    build_model,
    fast_hist,
    get_crop_size,
    get_device,
    get_ignore_label,
    get_image_stats,
    get_num_classes,
    load_checkpoint,
    load_config,
    per_class_iou,
    resolve_path,
    resize_long_side,
    save_visualization,
)


def cal_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    return fast_hist(a, b, n)


def resize_long(image: Image.Image, long_size: int = 513) -> Image.Image:
    return resize_long_side(image, long_size, Image.BILINEAR)


def pre_process(config: Dict, image: Image.Image, crop_size: int = 513) -> tuple[np.ndarray, int, int]:
    image = resize_long(image, crop_size)
    image_np = np.array(image.convert("RGB"), dtype=np.float32)
    resize_h, resize_w, _ = image_np.shape

    image_mean, image_std = get_image_stats(config)
    image_np = image_np[:, :, ::-1]
    image_np = (image_np - np.array(image_mean, dtype=np.float32)) / np.array(image_std, dtype=np.float32)

    pad_h = crop_size - resize_h
    pad_w = crop_size - resize_w
    if pad_h > 0 or pad_w > 0:
        image_np = np.pad(image_np, ((0, max(pad_h, 0)), (0, max(pad_w, 0)), (0, 0)), mode="constant")

    image_np = image_np.transpose((2, 0, 1))
    return image_np.astype(np.float32), resize_h, resize_w


@torch.no_grad()
def eval_batch(config: Dict, eval_net, img_lst: List[Image.Image], crop_size: int = 513, flip: bool = True) -> List[np.ndarray]:
    device = next(eval_net.parameters()).device
    batch_size = len(img_lst)
    batch_img = np.zeros((batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw: List[List[int]] = []

    for idx, image in enumerate(img_lst):
        img_np, resize_h, resize_w = pre_process(config, image, crop_size)
        batch_img[idx] = img_np
        resize_hw.append([resize_h, resize_w])

    batch_tensor = torch.from_numpy(np.ascontiguousarray(batch_img)).to(device)
    net_out = torch.softmax(eval_net(batch_tensor), dim=1)

    if flip:
        flipped_tensor = torch.flip(batch_tensor, dims=[3])
        net_out = net_out + torch.flip(torch.softmax(eval_net(flipped_tensor), dim=1), dims=[3])

    net_out = net_out.detach().cpu().numpy()
    result_lst: List[np.ndarray] = []
    for idx in range(batch_size):
        probs = net_out[idx][:, : resize_hw[idx][0], : resize_hw[idx][1]]
        ori_h, ori_w = img_lst[idx].size[1], img_lst[idx].size[0]
        probs_tensor = torch.from_numpy(probs).unsqueeze(0)
        probs_tensor = F.interpolate(probs_tensor, size=(ori_h, ori_w), mode="bilinear", align_corners=True)
        probs = probs_tensor.squeeze(0).permute(1, 2, 0).numpy()
        result_lst.append(probs)
    return result_lst


@torch.no_grad()
def eval_batch_scales(
    config: Dict,
    eval_net,
    img_lst: List[Image.Image],
    scales: List[float],
    base_crop_size: int = 513,
    flip: bool = True,
) -> List[np.ndarray]:
    sizes = [int((base_crop_size - 1) * scale) + 1 for scale in scales]
    probs_lst = eval_batch(config, eval_net, img_lst, crop_size=sizes[0], flip=flip)
    for crop_size in sizes[1:]:
        probs_lst_tmp = eval_batch(config, eval_net, img_lst, crop_size=crop_size, flip=flip)
        for idx in range(len(probs_lst)):
            probs_lst[idx] += probs_lst_tmp[idx]
    return [probs.argmax(axis=2).astype(np.int64) for probs in probs_lst]


@torch.no_grad()
def evaluate_model(
    config: Dict,
    model=None,
    voc_root: Optional[Path] = None,
    save_visualizations: bool = True,
    metrics_filename: str = "eval_metrics.json",
) -> Dict:
    base_dir = Path(__file__).resolve().parents[1]
    if model is None:
        device = get_device()
        model = build_model(config, phase="eval").to(device)
        checkpoint_path = resolve_path(base_dir, Path(config["paths"]["checkpoint_dir"]) / config["paths"]["checkpoint_name"])
        load_checkpoint(model, checkpoint_path, device)
    else:
        device = next(model.parameters()).device

    model.eval()
    if voc_root is None:
        metrics_path = resolve_path(base_dir, Path(config["paths"]["output_dir"]) / "train_metrics.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            voc_root = Path(metrics["voc_root"])
        else:
            data_root = Path(config.get("data", {}).get("root", "data"))
            voc_root = resolve_path(base_dir, data_root / "VOCdevkit" / "VOC2012")

    eval_cfg = config.get("eval", {})
    crop_size = get_crop_size(config)
    scales = [float(x) for x in eval_cfg.get("scales", [1.0])]
    flip = bool(eval_cfg.get("flip", True))
    ignore_label = get_ignore_label(config)
    num_classes = get_num_classes(config)

    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    eval_ids = get_eval_ids(voc_root)
    print(f"Evaluating {len(eval_ids)} validation images with scales={scales} flip={flip}", flush=True)

    vis_dir = resolve_path(base_dir, config["paths"]["visualization_dir"])
    saved = 0
    batch_size = int(eval_cfg.get("batch_size", 1))
    batch_img_lst: List[Image.Image] = []
    batch_msk_lst: List[np.ndarray] = []

    for index, image_id in enumerate(eval_ids, start=1):
        image, target = load_eval_sample(voc_root, image_id)
        batch_img_lst.append(image)
        batch_msk_lst.append(target)

        if len(batch_img_lst) == batch_size:
            batch_res = eval_batch_scales(config, model, batch_img_lst, scales=scales, base_crop_size=crop_size, flip=flip)
            for item_idx, prediction in enumerate(batch_res):
                target_eval = batch_msk_lst[item_idx].copy()
                target_eval[target_eval == ignore_label] = -1
                hist += cal_hist(target_eval.flatten(), prediction.flatten(), num_classes)

                if save_visualizations and saved < int(eval_cfg.get("num_visualizations", 3)):
                    vis_target = batch_msk_lst[item_idx].copy()
                    vis_target[vis_target == ignore_label] = 0
                    save_visualization(batch_img_lst[item_idx], prediction, vis_target, vis_dir / f"sample_{saved + 1}.png")
                    saved += 1

            batch_img_lst = []
            batch_msk_lst = []
            if index % 100 == 0:
                print(f"processed {index}/{len(eval_ids)} images", flush=True)

    if batch_img_lst:
        batch_res = eval_batch_scales(config, model, batch_img_lst, scales=scales, base_crop_size=crop_size, flip=flip)
        for item_idx, prediction in enumerate(batch_res):
            target_eval = batch_msk_lst[item_idx].copy()
            target_eval[target_eval == ignore_label] = -1
            hist += cal_hist(target_eval.flatten(), prediction.flatten(), num_classes)

            if save_visualizations and saved < int(eval_cfg.get("num_visualizations", 3)):
                vis_target = batch_msk_lst[item_idx].copy()
                vis_target[vis_target == ignore_label] = 0
                save_visualization(batch_img_lst[item_idx], prediction, vis_target, vis_dir / f"sample_{saved + 1}.png")
                saved += 1

    print(f"processed {len(eval_ids)}/{len(eval_ids)} images", flush=True)
    ious = per_class_iou(hist)
    miou = float(np.nanmean(ious))
    pixel_accuracy = float(np.diag(hist).sum() / hist.sum())
    class_accuracy = np.divide(
        np.diag(hist),
        hist.sum(axis=1),
        out=np.full(num_classes, np.nan),
        where=hist.sum(axis=1) != 0,
    )
    mean_class_accuracy = float(np.nanmean(class_accuracy))
    results = {
        "miou": miou,
        "pixel_accuracy": pixel_accuracy,
        "mean_class_accuracy": mean_class_accuracy,
        "class_iou": {VOC_CLASSES[i]: float(ious[i]) for i in range(num_classes)},
        "num_images": len(eval_ids),
    }

    output_dir = resolve_path(base_dir, config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / metrics_filename).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean IoU {miou:.6f}", flush=True)
    return results


def main() -> None:
    config = load_config("config.json")
    evaluate_model(config)


if __name__ == "__main__":
    main()
