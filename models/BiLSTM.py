import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
    
    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.embedding(src)                      # [batch, src_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)       # outputs: [batch, src_len, 2*hidden]
        return outputs, hidden, cell                        # hidden/cell: [2, batch, hidden]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch, hidden]
        # encoder_outputs: [batch, src_len, 2*hidden]
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)       # [batch, src_len, hidden]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hidden]
        attention = self.v(energy).squeeze(2)                    # [batch, src_len]
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)                       # [batch, src_len]

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, pad_idx, attention):
        super().__init__()
        self.decoder_hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM((2 * hidden_dim) + emb_dim, hidden_dim, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim + 2 * hidden_dim, output_dim)  # 512 + 1024 = 1536 → output_dim
        self.attention = attention

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # input: [batch]
        input = input.unsqueeze(1)                             # [batch, 1]
        embedded = self.embedding(input)                       # [batch, 1, emb_dim]

        # hidden: [1, batch, hidden_dim] → decoder_hidden = [batch, hidden_dim]
        decoder_hidden = hidden[-1]                            # [batch, hidden_dim]
        attn_weights = self.attention(decoder_hidden, encoder_outputs, mask)  # [batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)               # [batch, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)     # [batch, 1, 2*hidden_dim]

        rnn_input = torch.cat((embedded, context), dim=2)      # [batch, 1, emb + 2*hidden]
        output, (hidden, cell) = self.lstm(
            rnn_input, 
            (hidden.contiguous(), cell.contiguous())  # ✅ 加这一步
        )

        output = output.squeeze(1)                             # [batch, hidden_dim]
        context = context.squeeze(1)                           # [batch, 2*hidden_dim]
        output = self.fc_out(torch.cat((output, context), dim=1))  # [batch, output_dim]
        return output, hidden, cell, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def create_mask(self, src):
        return (src != self.pad_idx).to(self.device)  # [batch, src_len]

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len], trg: [batch, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)  # hidden: [2, batch, hidden_dim]

        # 改：将 encoder 的 forward & backward hidden → 映射为 decoder 用的 shape（512）
        hidden = torch.tanh(torch.cat((hidden[0:1], hidden[1:2]), dim=2))  # [1, batch, 1024]
        cell = torch.tanh(torch.cat((cell[0:1], cell[1:2]), dim=2))        # [1, batch, 1024]

        # 再降维到 decoder hidden_dim（512）：
        hidden = hidden[:, :, :self.decoder.decoder_hidden_dim]  # → [1, batch, 512]
        cell = cell[:, :, :self.decoder.decoder_hidden_dim]      # → [1, batch, 512]

        input = trg[:, 0]  # start token
        mask = self.create_mask(src)

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

