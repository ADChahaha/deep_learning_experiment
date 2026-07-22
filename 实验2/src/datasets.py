from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation

from .utils import get_crop_size, get_ignore_label, get_image_stats, normalize_bgr_image


class SegmentationTrainDataset(Dataset):
    def __init__(
        self,
        voc_root: Path,
        crop_size: int,
        min_scale: float,
        max_scale: float,
        ignore_label: int,
        image_mean: List[float],
        image_std: List[float],
    ) -> None:
        self.voc_root = voc_root
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ignore_label = ignore_label
        self.image_mean = image_mean
        self.image_std = image_std
        self.image_dir = voc_root / "JPEGImages"
        self.mask_dir = voc_root / "SegmentationClass"
        split_file = voc_root / "ImageSets" / "Segmentation" / "train.txt"
        self.ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def _random_scale(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        scale = float(np.random.uniform(self.min_scale, self.max_scale))
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))
        image = image.resize((new_width, new_height), Image.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.NEAREST)
        return image, mask

    def _pad(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        pad_width = max(self.crop_size - image.width, 0)
        pad_height = max(self.crop_size - image.height, 0)
        if pad_width > 0 or pad_height > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=self.ignore_label)
        return image, mask

    def _random_crop(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        max_left = image.width - self.crop_size
        max_top = image.height - self.crop_size
        left = 0 if max_left <= 0 else int(np.random.randint(0, max_left + 1))
        top = 0 if max_top <= 0 else int(np.random.randint(0, max_top + 1))
        box = (left, top, left + self.crop_size, top + self.crop_size)
        return image.crop(box), mask.crop(box)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[index]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{image_id}.png")

        image, mask = self._random_scale(image, mask)
        image, mask = self._pad(image, mask)
        image, mask = self._random_crop(image, mask)

        if float(np.random.rand()) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_np = np.array(image, dtype=np.float32)
        image_np = normalize_bgr_image(image_np, self.image_mean, self.image_std)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        mask_array = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor


def ensure_voc_dataset(data_root: Path, year: str = "2012", download: bool = True) -> Path:
    VOCSegmentation(root=str(data_root), year=year, image_set="train", download=download)
    VOCSegmentation(root=str(data_root), year=year, image_set="val", download=False)
    return data_root / "VOCdevkit" / f"VOC{year}"


def create_train_loader(config, base_dir: Path) -> Tuple[DataLoader, Path]:
    data_cfg = config.get("data", {})
    data_root = base_dir / data_cfg.get("root", "data")
    voc_root = ensure_voc_dataset(
        data_root=data_root,
        year=str(data_cfg.get("year", "2012")),
        download=bool(data_cfg.get("download", True)),
    )
    image_mean, image_std = get_image_stats(config)
    dataset = SegmentationTrainDataset(
        voc_root=voc_root,
        crop_size=get_crop_size(config),
        min_scale=float(data_cfg.get("min_scale", config.get("min_scale", 0.5))),
        max_scale=float(data_cfg.get("max_scale", config.get("max_scale", 2.0))),
        ignore_label=get_ignore_label(config),
        image_mean=image_mean,
        image_std=image_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", config.get("batch_size", 16))),
        shuffle=True,
        drop_last=True,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, voc_root


def get_eval_ids(voc_root: Path) -> List[str]:
    split_file = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_eval_sample(voc_root: Path, image_id: str) -> Tuple[Image.Image, np.ndarray]:
    image = Image.open(voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
    mask = np.array(Image.open(voc_root / "SegmentationClass" / f"{image_id}.png"), dtype=np.int64)
    return image, mask
