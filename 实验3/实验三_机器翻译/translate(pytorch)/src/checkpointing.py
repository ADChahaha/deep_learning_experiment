import json
import os
import tempfile

import torch


def save_model(obj, path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def append_json_record(record, path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_epoch_checkpoint(checkpoint_dir, epoch, payload):
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch-{epoch:03d}.pt")
    save_model(payload, checkpoint_path)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    save_model(payload, latest_path)
    return checkpoint_path
