#!/usr/bin/env python3
"""Run PFNet on paper-style samples and build report-ready comparison figures."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
PFNET_DIR = ROOT / "CVPR2021_PFNet"
TEST_ZIP = PFNET_DIR / "TestDataset.zip"
OUTPUT_DIR = ROOT / "report_assets" / "sample_inference"

PAPER_FIGURE4_SAMPLES = [
    ("COD10K", "COD10K-CAM-1-Aquatic-13-Pipefish-554"),
    ("COD10K", "COD10K-CAM-1-Aquatic-13-Pipefish-614"),
    ("COD10K", "COD10K-CAM-1-Aquatic-6-Fish-186"),
    ("COD10K", "COD10K-CAM-2-Terrestrial-23-Cat-1328"),
    ("COD10K", "COD10K-CAM-3-Flying-53-Bird-3196"),
    ("COD10K", "COD10K-CAM-3-Flying-64-Moth-4467"),
    ("COD10K", "COD10K-CAM-1-Aquatic-18-StarFish-1178"),
    ("COD10K", "COD10K-CAM-1-Aquatic-15-SeaHorse-1003"),
    ("COD10K", "COD10K-CAM-4-Amphibian-68-Toad-4964"),
]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_image_from_zip(zf: zipfile.ZipFile, path: str, mode: str) -> Image.Image:
    with zf.open(path) as f:
        return Image.open(BytesIO(f.read())).convert(mode)


def normalize_prediction(prediction: torch.Tensor) -> np.ndarray:
    pred = prediction.squeeze().detach().float().cpu().numpy()
    pred = pred - pred.min()
    denom = pred.max() + 1e-8
    return (pred / denom * 255).astype(np.uint8)


def mask_iou(pred: Image.Image, gt: Image.Image) -> float:
    pred_np = np.array(pred.convert("L")) >= 128
    gt_np = np.array(gt.convert("L")) >= 128
    union = np.logical_or(pred_np, gt_np).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_np, gt_np).sum() / union)


def mask_mae(pred: Image.Image, gt: Image.Image) -> float:
    pred_np = np.array(pred.convert("L"), dtype=np.float32) / 255.0
    gt_np = np.array(gt.convert("L"), dtype=np.float32) / 255.0
    return float(np.abs(pred_np - gt_np).mean())


def try_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def overlay_mask(image: Image.Image, mask: Image.Image, alpha: int = 175, threshold: int = 40) -> Image.Image:
    base = image.convert("RGBA")
    mask_np = np.array(mask.convert("L"))
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha_layer = Image.fromarray(np.where(mask_np > threshold, alpha, 0).astype(np.uint8), mode="L")
    red.putalpha(alpha_layer)
    return Image.alpha_composite(base, red).convert("RGB")


def resize_tile(image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(tile_size, Image.Resampling.BILINEAR)


def make_paper_style_grid(rows: list[dict[str, Image.Image]], output_path: Path) -> None:
    tile_w, tile_h = 164, 112
    gap = 5
    margin_x = 20
    margin_top = 18
    label_h = 38
    labels = ["Image", "GT", "Ours"]
    width = margin_x * 2 + len(labels) * tile_w + (len(labels) - 1) * gap
    height = margin_top + len(rows) * tile_h + (len(rows) - 1) * gap + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = try_font(25)

    for row_idx, row in enumerate(rows):
        y = margin_top + row_idx * (tile_h + gap)
        for col_idx, label in enumerate(labels):
            x = margin_x + col_idx * (tile_w + gap)
            canvas.paste(resize_tile(row[label], (tile_w, tile_h)), (x, y))

    label_y = margin_top + len(rows) * tile_h + (len(rows) - 1) * gap + 6
    for col_idx, label in enumerate(labels):
        x = margin_x + col_idx * (tile_w + gap)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x + (tile_w - (bbox[2] - bbox[0])) / 2, label_y), label, fill=(0, 0, 0), font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--scale", type=int, default=416)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    sys.path.insert(0, str(PFNET_DIR))
    from PFNet import PFNet  # noqa: PLC0415

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.scale, args.scale)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    original_torch_load = torch.load

    def trusted_legacy_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_legacy_load
    try:
        net = PFNet(str(PFNET_DIR / "resnet50-19c8e357.pth")).to(device)
    finally:
        torch.load = original_torch_load
    state = torch.load(PFNET_DIR / "PFNet.pth", map_location=device, weights_only=False)
    net.load_state_dict(state)
    net.eval()

    figure_rows = []
    stats = []
    with zipfile.ZipFile(TEST_ZIP) as zf, torch.no_grad():
        for dataset, sample in PAPER_FIGURE4_SAMPLES:
            image_path = f"TestDataset/{dataset}/Imgs/{sample}.jpg"
            gt_path = f"TestDataset/{dataset}/GT/{sample}.png"
            image = load_image_from_zip(zf, image_path, "RGB")
            gt = load_image_from_zip(zf, gt_path, "L")

            input_tensor = transform(image).unsqueeze(0).to(device)
            start = time.perf_counter()
            _, _, _, prediction = net(input_tensor)
            elapsed = time.perf_counter() - start

            pred_np = normalize_prediction(prediction)
            pred_img = Image.fromarray(pred_np, mode="L").resize(image.size, Image.Resampling.BILINEAR)
            pred_path = pred_dir / f"{dataset}_{sample}.png"
            pred_img.save(pred_path)

            figure_rows.append(
                {
                    "Image": image,
                    "GT": overlay_mask(image, gt, alpha=175, threshold=128),
                    "Ours": overlay_mask(image, pred_img, alpha=175, threshold=40),
                }
            )
            stats.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "width": image.width,
                    "height": image.height,
                    "device": str(device),
                    "seconds": f"{elapsed:.4f}",
                    "mae": f"{mask_mae(pred_img, gt):.4f}",
                    "iou_at_0.5": f"{mask_iou(pred_img, gt):.4f}",
                    "prediction": str(pred_path.relative_to(ROOT)),
                }
            )

    figure_path = args.output_dir / "pfnet_figure4_style.png"
    make_paper_style_grid(figure_rows, figure_path)

    csv_path = args.output_dir / "sample_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    print(f"device={device}")
    print(f"samples={len(stats)}")
    print(f"figure={figure_path}")
    print(f"metrics={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
