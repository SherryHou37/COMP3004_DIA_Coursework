import torch
from torch.utils.data import Dataset

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_len=150, max_target_len=150):
        self.tokenizer = tokenizer
        self.input_seqs = tokenizer.texts_to_sequences(input_texts)
        self.target_seqs = tokenizer.texts_to_sequences(target_texts)

        self.input_seqs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq[:max_len]) for seq in self.input_seqs],
            batch_first=True, padding_value=0
        )
        self.target_seqs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq[:max_target_len]) for seq in self.target_seqs],
            batch_first=True, padding_value=0
        )

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return self.input_seqs[idx], self.target_seqs[idx]
