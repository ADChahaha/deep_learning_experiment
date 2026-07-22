from collections import Counter
from math import exp, log


PAD_ID = 0
SOS_ID = 1


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def decode_english(token_ids, vocab):
    tokens = []
    for token_id in token_ids:
        if token_id == PAD_ID:
            break
        if token_id == SOS_ID:
            continue
        tokens.append(vocab[token_id])
    return " ".join(tokens)


def decode_chinese(token_ids, vocab):
    chars = []
    for token_id in token_ids:
        if token_id == PAD_ID:
            break
        if token_id == SOS_ID:
            continue
        chars.append(vocab[token_id])
    return "".join(chars)


def _extract_ngrams(tokens, order):
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[idx : idx + order]) for idx in range(len(tokens) - order + 1))


def corpus_bleu(references, hypotheses, max_order=4, smooth=True):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    hypothesis_length = 0

    for reference, hypothesis in zip(references, hypotheses):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)

        for order in range(1, max_order + 1):
            reference_ngrams = _extract_ngrams(reference, order)
            hypothesis_ngrams = _extract_ngrams(hypothesis, order)
            overlap = hypothesis_ngrams & reference_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(len(hypothesis) - order + 1, 0)

    precisions = [0.0] * max_order
    for idx in range(max_order):
        if smooth:
            precisions[idx] = (matches_by_order[idx] + 1.0) / (possible_matches_by_order[idx] + 1.0)
        elif possible_matches_by_order[idx] > 0:
            precisions[idx] = matches_by_order[idx] / possible_matches_by_order[idx]

    if min(precisions) > 0:
        geo_mean = exp(sum(log(precision) for precision in precisions) / max_order)
    else:
        geo_mean = 0.0

    if hypothesis_length == 0:
        brevity_penalty = 0.0
    elif hypothesis_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = exp(1.0 - float(reference_length) / float(hypothesis_length))

    return {
        "bleu": 100.0 * geo_mean * brevity_penalty,
        "brevity_penalty": brevity_penalty,
        "precisions": [100.0 * precision for precision in precisions],
        "reference_length": reference_length,
        "hypothesis_length": hypothesis_length,
    }
