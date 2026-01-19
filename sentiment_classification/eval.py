from utils import load_glove
import argparse
from modelmodule import MyModel
import torch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("text", type=str)
arg_parser.add_argument("--embedding_path", type=str, default="data/glove.6B.100d.txt")
arg_parser.add_argument("--model_path", type=str, default="checkpoint/best.ckpt")


args = arg_parser.parse_args()

if __name__ == "__main__":
    vocab, embedding_tensor = load_glove(args.embedding_path)
    vocab
    model = MyModel.load_from_checkpoint(args.model_path, embedding_path=args.embedding_path)
    model.eval()
    # 处理文本
    text = args.text
    text = text.lower().split()
    ids = [vocab.get(word, vocab["<unk>"]) for word in text]
    ids_tensor = torch.tensor(ids)
    ids_tensor.unsqueeze_(dim=0)
    y = model.model(ids_tensor, torch.tensor([len(ids)]))
    preds = torch.sigmoid(y) > 0.5
    if preds.item() == 0:
        print("Negative")
    else:
        print("Positive")

    