import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

DEFAULT_IMAGE_MEAN = [103.53, 116.28, 123.675]
DEFAULT_IMAGE_STD = [57.375, 57.120, 58.395]


def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def get_num_classes(config: Dict[str, Any]) -> int:
    return int(get_nested(config, "model", "num_classes", default=config.get("num_classes", 21)))


def get_ignore_label(config: Dict[str, Any]) -> int:
    return int(get_nested(config, "data", "ignore_label", default=config.get("ignore_label", 255)))


def get_crop_size(config: Dict[str, Any]) -> int:
    return int(get_nested(config, "data", "crop_size", default=config.get("crop_size", 513)))


def get_model_name(config: Dict[str, Any]) -> str:
    name = get_nested(config, "model", "name", default=config.get("model", "deeplab_v3_s8"))
    if name in {"deeplabv3_resnet101", "deeplabv3_resnet50"}:
        return "deeplab_v3_s8"
    return str(name)


def get_image_stats(config: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    mean = get_nested(config, "data", "image_mean", default=config.get("image_mean", DEFAULT_IMAGE_MEAN))
    std = get_nested(config, "data", "image_std", default=config.get("image_std", DEFAULT_IMAGE_STD))
    return [float(x) for x in mean], [float(x) for x in std]


def freeze_batchnorm_layers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.BatchNorm2d):
            child.eval()
            for parameter in child.parameters():
                parameter.requires_grad = False


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rate: int) -> None:
        super().__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=atrous_rate,
                dilation=atrous_rate,
                bias=False,
            )
        self.block = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.pool(x)
        out = self.block(out)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=True)


class ASPP(nn.Module):
    def __init__(self, atrous_rates: List[int], phase: str = "train", in_channels: int = 2048, num_classes: int = 21) -> None:
        super().__init__()
        out_channels = 256
        self.phase = phase
        self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0])
        self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1])
        self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2])
        self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3])
        self.aspp_pooling = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [
            self.aspp1(x),
            self.aspp2(x),
            self.aspp3(x),
            self.aspp4(x),
            self.aspp_pooling(x),
        ]
        out = torch.cat(outputs, dim=1)
        out = self.project(out)
        if self.phase == "train":
            out = self.dropout(out)
        return self.classifier(out)


class DeepLabV3(nn.Module):
    def __init__(
        self,
        phase: str = "train",
        num_classes: int = 21,
        output_stride: int = 8,
        freeze_bn: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if output_stride == 16:
            dilation = [False, False, True]
        elif output_stride == 8:
            dilation = [False, True, True]
        else:
            raise ValueError(f"Unsupported output_stride: {output_stride}")

        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet101(weights=weights, replace_stride_with_dilation=dilation)

        self.backbone_initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aspp = ASPP([1, 6, 12, 18], phase=phase, in_channels=2048, num_classes=num_classes)

        if freeze_bn:
            freeze_batchnorm_layers(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        x = self.backbone_initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)


def build_model(config: Dict[str, Any], phase: str = "train", load_pretrained_checkpoint: bool = True) -> nn.Module:
    model_name = get_model_name(config)
    num_classes = get_num_classes(config)
    freeze_bn = bool(get_nested(config, "model", "freeze_bn", default=config.get("freeze_bn", True)))
    pretrained = bool(get_nested(config, "model", "pretrained", default=config.get("pretrained", True)))

    if model_name == "deeplab_v3_s16":
        model = DeepLabV3(phase=phase, num_classes=num_classes, output_stride=16, freeze_bn=freeze_bn, pretrained=pretrained)
    elif model_name == "deeplab_v3_s8":
        model = DeepLabV3(phase=phase, num_classes=num_classes, output_stride=8, freeze_bn=freeze_bn, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    checkpoint_value = get_nested(config, "model", "pretrained_checkpoint", default=config.get("ckpt_file", ""))
    if load_pretrained_checkpoint and checkpoint_value:
        checkpoint_path = resolve_path(Path(__file__).resolve().parents[1], str(checkpoint_value))
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "model_state" in state_dict:
                state_dict = state_dict["model_state"]
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained checkpoint: {checkpoint_path}")
        else:
            print(f"Pretrained checkpoint not found, skip loading: {checkpoint_path}")
    return model


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, num_cls: int = 21, ignore_label: int = 255) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, labels.long())


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return criterion(logits, targets)


def cosine_lr(base_lr: float, decay_steps: int, total_steps: int) -> List[float]:
    values: List[float] = []
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        values.append(base_lr * 0.5 * (1 + math.cos(math.pi * step_ / max(decay_steps, 1))))
    return values


def poly_lr(base_lr: float, decay_steps: int, total_steps: int, end_lr: float = 0.0001, power: float = 0.9) -> List[float]:
    values: List[float] = []
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        values.append((base_lr - end_lr) * ((1.0 - step_ / max(decay_steps, 1)) ** power) + end_lr)
    return values


def exponential_lr(base_lr: float, decay_steps: int, decay_rate: float, total_steps: int, staircase: bool = False) -> List[float]:
    values: List[float] = []
    for i in range(total_steps):
        power_ = i // decay_steps if staircase else float(i) / max(decay_steps, 1)
        values.append(base_lr * (decay_rate**power_))
    return values


def build_lr_schedule(config: Dict[str, Any], total_steps: int) -> List[float]:
    train_cfg = config.get("train", {})
    lr_type = str(train_cfg.get("lr_type", config.get("lr_type", "cos")))
    base_lr = float(train_cfg.get("base_lr", train_cfg.get("lr", config.get("base_lr", 1e-3))))
    if base_lr <= 0:
        base_lr = float(train_cfg.get("lr", 1e-3))
    decay_steps = int(train_cfg.get("lr_decay_step", config.get("lr_decay_step", total_steps)))
    decay_rate = float(train_cfg.get("lr_decay_rate", config.get("lr_decay_rate", 0.1)))

    if lr_type == "cos":
        return cosine_lr(base_lr, total_steps, total_steps)
    if lr_type == "poly":
        return poly_lr(base_lr, total_steps, total_steps, end_lr=0.0, power=0.9)
    if lr_type == "exp":
        return exponential_lr(base_lr, decay_steps, decay_rate, total_steps, staircase=True)
    raise ValueError(f"Unknown learning rate type: {lr_type}")


def save_checkpoint(
    model: nn.Module,
    config: Dict[str, Any],
    checkpoint_path: Path,
    history: Dict[str, List[float]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "history": history,
            "classes": VOC_CLASSES,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def get_voc_palette() -> List[int]:
    palette = [0] * (256 * 3)
    for class_index in range(256):
        label = class_index
        red = green = blue = 0
        bit = 0
        while label:
            red |= ((label >> 0) & 1) << (7 - bit)
            green |= ((label >> 1) & 1) << (7 - bit)
            blue |= ((label >> 2) & 1) << (7 - bit)
            bit += 1
            label >>= 3
        palette[class_index * 3 + 0] = red
        palette[class_index * 3 + 1] = green
        palette[class_index * 3 + 2] = blue
    return palette


def colorize_mask(mask: np.ndarray) -> Image.Image:
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette(get_voc_palette())
    return image.convert("RGB")


def plot_training_curve(losses: Iterable[float], output_path: Path) -> None:
    values = list(losses)
    if not values:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def fast_hist(label_true: np.ndarray, label_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (label_true >= 0) & (label_true < num_classes)
    return np.bincount(
        num_classes * label_true[mask].astype(np.int64) + label_pred[mask].astype(np.int64),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def per_class_iou(hist: np.ndarray) -> np.ndarray:
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.divide(
        np.diag(hist),
        denominator,
        out=np.zeros(hist.shape[0], dtype=np.float64),
        where=denominator != 0,
    )


def resize_long_side(image: Image.Image, long_size: int, interpolation: int) -> Image.Image:
    width, height = image.size
    if height > width:
        new_height = long_size
        new_width = int(1.0 * long_size * width / height)
    else:
        new_width = long_size
        new_height = int(1.0 * long_size * height / width)
    return image.resize((new_width, new_height), interpolation)


def normalize_bgr_image(image_np: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    image_bgr = image_np[:, :, ::-1].astype(np.float32)
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)
    return (image_bgr - mean_np) / std_np


def save_visualization(
    image: Image.Image,
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
) -> None:
    pred_rgb = np.array(colorize_mask(prediction))
    target_rgb = np.array(colorize_mask(target))
    image_np = np.array(image.convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(image_np)
    axes[1].imshow(pred_rgb, alpha=0.65)
    axes[1].set_title("Prediction")
    axes[2].imshow(image_np)
    axes[2].imshow(target_rgb, alpha=0.65)
    axes[2].set_title("Ground Truth")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
