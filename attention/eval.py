import torch
import re
from nltk.translate.bleu_score import corpus_bleu
from modelmodule import MyModel
from utils import LoadVocab
from dataset import Multi30K

# ======================
def tokenize(text: str):
    text = text.rstrip()
    return [tok.lower() for tok in re.findall(r"\w+|[^\w\s]", text)]


# ======================
# 加载 vocab & model
# ======================
de_vocab = LoadVocab(
    "assets/vocabs/de_vocab.json",
    "assets/vocabs/de_idx_vocab.json"
)
en_vocab = LoadVocab(
    "assets/vocabs/en_vocab.json",
    "assets/vocabs/en_idx_vocab.json"
)

model = MyModel.load_from_checkpoint(
    "checkpoint/best.ckpt",
    embedding_dim=256,
    nhead=8,
    encoder_num_layers=3,
    decoder_num_layers=3,
)
model.eval()
model.to("cpu")


# ======================
@torch.no_grad()
def translate(src_sentence: str):
    # tokenize source
    src_tokens = tokenize(src_sentence)

    # ids → tensor
    src_ids = de_vocab.encode(src_tokens)
    src_tensor = torch.tensor(src_ids).unsqueeze(0)

    # padding mask
    src_len = src_tensor.size(1)
    src_key_padding_mask = torch.zeros(
        (1, src_len), dtype=torch.bool
    )

    # generate
    output_ids = model.model.generate(
        src_tensor, src_key_padding_mask
    )

    output_ids = output_ids.squeeze(0).tolist()

    # remove <bos>
    if output_ids and output_ids[0] == en_vocab.bos_idx:
        output_ids = output_ids[1:]

    # truncate at <eos>
    if en_vocab.eos_idx in output_ids:
        output_ids = output_ids[:output_ids.index(en_vocab.eos_idx)]

    # id → str → token
    output_ids = [str(i) for i in output_ids]
    tokens = en_vocab.decode(output_ids)

    # filter special tokens
    tokens = [
        tok for tok in tokens
        if tok not in ("<bos>", "<eos>", "<pad>", "<unk>")
    ]

    return tokens


def evaluate_bleu(dataset, max_samples=100):
    references = []
    hypotheses = []

    for i, (de, en) in enumerate(dataset):
        if i >= max_samples:
            break

        de_ids = [str(x) for x in de.tolist()]
        de_tokens = de_vocab.decode(de_ids)
        src_sentence = " ".join(de_tokens)

        en_ids = [str(x) for x in en.tolist()]
        ref_tokens = en_vocab.decode(en_ids)
        ref_tokens = [
            tok for tok in ref_tokens
            if tok not in ("<bos>", "<eos>", "<pad>", "<unk>")
        ]

        pred_tokens = translate(src_sentence)

        references.append([ref_tokens])   
        hypotheses.append(pred_tokens)

        if i < 5:
            print("=" * 50)
            print("SRC :", src_sentence)
            print("REF :", " ".join(ref_tokens))
            print("PRED:", " ".join(pred_tokens))

    bleu = corpus_bleu(references, hypotheses)
    return bleu


if __name__ == "__main__":
    dataset = Multi30K(
        "assets/datasets/test/",
        de_vocab,
        en_vocab
    )

    bleu = evaluate_bleu(dataset, max_samples=100)
    print(f"\nBLEU score: {bleu:.4f}")
