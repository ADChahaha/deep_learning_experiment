import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Model(nn.Module):
    def __init__(
        self,
        embedding_tensor,
        pad_idx,
        input_size: int = 100,
        hidden_size=256,
        num_layers=1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor, freeze=True, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1))
        self.dropout = nn.Dropout()

    def forward(self, x, lengths):

        x = self.embedding(x)
        x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]

        # x = self.dropout(last_hidden)
        x = last_hidden
        logits = self.fc(x)
        return logits


if __name__ == "__main__":
    pass
