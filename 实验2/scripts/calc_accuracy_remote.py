import json
import sys
from pathlib import Path

import numpy as np

from src.datasets import get_eval_ids, load_eval_sample
from src.evaluate import cal_hist, eval_batch_scales
from src.utils import build_model, get_device, load_checkpoint


def main() -> None:
    base = Path(sys.argv[1]).resolve()
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    device = get_device()
    model = build_model(config, phase="eval").to(device)
    checkpoint = base / "checkpoints" / "deeplabv3_resnet101_voc2012.pth"
    load_checkpoint(model, checkpoint, device)
    model.eval()

    voc_root = Path(config["data"]["root"]) / "VOCdevkit" / "VOC2012"
    eval_ids = get_eval_ids(voc_root)
    num_classes = int(config.get("model", {}).get("num_classes", 21))
    ignore_label = int(config.get("data", {}).get("ignore_label", 255))
    scales = [float(x) for x in config.get("eval", {}).get("scales", [1.0])]
    flip = bool(config.get("eval", {}).get("flip", True))
    crop_size = int(config.get("data", {}).get("crop_size", 513))

    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    for image_id in eval_ids:
        image, target = load_eval_sample(voc_root, image_id)
        pred = eval_batch_scales(config, model, [image], scales=scales, base_crop_size=crop_size, flip=flip)[0]
        target_eval = target.copy()
        target_eval[target_eval == ignore_label] = -1
        hist += cal_hist(target_eval.flatten(), pred.flatten(), num_classes)

    pixel_acc = float(np.diag(hist).sum() / hist.sum())
    class_acc = np.divide(
        np.diag(hist),
        hist.sum(axis=1),
        out=np.full(num_classes, np.nan),
        where=hist.sum(axis=1) != 0,
    )
    mean_class_acc = float(np.nanmean(class_acc))
    print(json.dumps({"pixel_accuracy": pixel_acc, "mean_class_accuracy": mean_class_acc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
