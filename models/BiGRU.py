import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        if hidden.dim() == 3:      # [1, B, H] → [B, 1, H]
            hidden = hidden.permute(1, 0, 2)
        if hidden.dim() == 2:      # [B, H]   → [B, 1, H]
            hidden = hidden.unsqueeze(1)

        src_len = encoder_outputs.size(1)
        hidden = hidden.expand(-1, src_len, -1)      # [B, src_len, H]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attn = self.v(energy).squeeze(2)             # [B, src_len]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        return torch.softmax(attn, dim=1)


class EncoderBiGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)

        # 双向拼接，压缩为单向（2 * hidden_dim）
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch, 2 * hidden_dim]

        # print(f"Encoder Hidden Size: {hidden.size()}")

        return outputs, hidden

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, pad_idx, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.attention = attention

    def forward(self, input, hidden, encoder_outputs, mask=None):
        input = input.unsqueeze(1)                    # [B, 1]
        embedded = self.embedding(input)              # [B, 1, emb_dim]

        # 注意力
        attn_weights = self.attention(hidden, encoder_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)

        # 只在 hidden 还是 2‑D 时补一维
        if hidden.dim() == 2:                         # [B, H] -> [1, B, H]
            hidden = hidden.unsqueeze(0)

        output, hidden = self.gru(rnn_input, hidden)  # hidden 必定 3‑D
        prediction = self.fc_out(output.squeeze(1))   # [B, vocab]

        return prediction, hidden, attn_weights

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
        self.bridge = nn.Linear(encoder.gru.hidden_size * 2, decoder.gru.hidden_size)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        outputs = torch.zeros(batch_size, trg_len,
                              self.decoder.fc_out.out_features,
                              device=self.device)

        encoder_outputs, enc_hidden = self.encoder(src)       # [B, 2H]
        dec_hidden = torch.tanh(self.bridge(enc_hidden))      # [B, H]
        dec_hidden = dec_hidden.unsqueeze(0)                  # --> [1, B, H]

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, dec_hidden, _ = self.decoder(input, dec_hidden, encoder_outputs)
            outputs[:, t] = output
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else output.argmax(1)

        return outputs
