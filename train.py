#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import math


########################################
# Repro
########################################

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


########################################
# Dataset & vocab utilities
########################################

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>"]
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


def build_vocabs(train_df: pd.DataFrame):
    """
    Build character-level vocabularies for source (Hindi) and target (Roman)
    from the training data only.
    """
    all_src_text = "".join(train_df["native"].tolist())
    all_tgt_text = "".join(train_df["roman"].tolist())

    src_chars = sorted(list(set(all_src_text)))
    tgt_chars = sorted(list(set(all_tgt_text)))

    src_itos = SPECIAL_TOKENS + src_chars
    tgt_itos = SPECIAL_TOKENS + tgt_chars

    src_stoi = {ch: i for i, ch in enumerate(src_itos)}
    tgt_stoi = {ch: i for i, ch in enumerate(tgt_itos)}

    pad_idx = tgt_stoi[PAD_TOKEN]
    sos_idx = tgt_stoi[SOS_TOKEN]
    eos_idx = tgt_stoi[EOS_TOKEN]

    print(f"[INFO] Source vocab size: {len(src_itos)}")
    print(f"[INFO] Target vocab size: {len(tgt_itos)}")

    return src_stoi, tgt_stoi, src_itos, tgt_itos, pad_idx, sos_idx, eos_idx


def encode_src_word(word: str, src_stoi: dict):
    return [src_stoi[ch] for ch in word if ch in src_stoi]


def encode_tgt_word(word: str, tgt_stoi: dict, sos_idx: int, eos_idx: int):
    ids = [sos_idx]
    ids += [tgt_stoi[ch] for ch in word if ch in tgt_stoi]
    ids.append(eos_idx)
    return ids


class TransliterationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_stoi: dict, tgt_stoi: dict, sos_idx: int, eos_idx: int):
        self.src_words = df["native"].tolist()
        self.tgt_words = df["roman"].tolist()
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __len__(self):
        return len(self.src_words)

    def __getitem__(self, idx):
        src_word = self.src_words[idx]
        tgt_word = self.tgt_words[idx]

        src_ids = torch.tensor(encode_src_word(src_word, self.src_stoi), dtype=torch.long)
        tgt_ids = torch.tensor(encode_tgt_word(tgt_word, self.tgt_stoi, self.sos_idx, self.eos_idx),
                               dtype=torch.long)

        return src_ids, tgt_ids


def collate_fn(batch, pad_idx: int):
    src_seqs, tgt_seqs = zip(*batch)

    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)

    return src_padded, src_lens, tgt_padded, tgt_lens


########################################
# Model
########################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, n_heads, num_layers, ff_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, src):
        """
        src: (B, src_len)
        Returns:
          memory: (B, src_len, emb_dim)
          src_key_padding_mask: (B, src_len) bool
        """
        # (B, src_len, emb_dim)
        embedded = self.dropout(self.embedding(src))
        embedded = self.pos_encoder(embedded)

        src_key_padding_mask = (src == self.pad_idx)  # True where padding

        memory = self.encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask,
        )
        return memory, src_key_padding_mask


class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, n_heads, num_layers, ff_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(emb_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def _generate_square_subsequent_mask(self, sz, device):
        # standard causal mask: True above diagonal (masked)
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask  # (sz, sz)

    def forward(self, tgt_input, memory, memory_key_padding_mask):
        """
        tgt_input: (B, tgt_len_in)  [this is input sequence, typically shifted right]
        memory: (B, src_len, emb_dim)
        memory_key_padding_mask: (B, src_len)
        Returns:
          logits: (B, tgt_len_in, output_dim)
        """
        device = tgt_input.device
        tgt_key_padding_mask = (tgt_input == self.pad_idx)  # (B, tgt_len_in)

        # (B, tgt_len_in, emb_dim)
        embedded = self.dropout(self.embedding(tgt_input))
        embedded = self.pos_encoder(embedded)

        tgt_len = tgt_input.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)  # (tgt_len, tgt_len)

        dec_output = self.decoder(
            embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, tgt_len_in, emb_dim)

        logits = self.fc_out(dec_output)  # (B, tgt_len_in, output_dim)
        return logits


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.0):
        """
        src: (B, src_len)
        tgt: (B, tgt_len), includes <sos> at position 0, <eos> somewhere after.
        Returns:
          outputs: (B, tgt_len, vocab_size)
        """
        # Encode
        memory, src_key_padding_mask = self.encoder(src)  # memory: (B, src_len, D)

        # For training: we feed everything except last token as input,
        # and predict tokens 1..T-1.
        tgt_input = tgt[:, :-1]      # (B, T-1)
        logits = self.decoder(
            tgt_input,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, T-1, vocab)

        # We want outputs to have same length as tgt (so add a dummy zero at t=0)
        B, Tm1, V = logits.shape
        outputs = torch.zeros(B, Tm1 + 1, V, device=self.device)
        outputs[:, 1:, :] = logits

        return outputs


########################################
# Training & evaluation
########################################

def train_one_epoch(model, loader, optimizer, criterion, pad_idx, teacher_forcing_ratio=0.5, device="cpu"):
    model.train()
    epoch_loss = 0.0

    for src, src_lens, tgt, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        outputs = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        # shift: ignore t=0 (all zeros initially)
        output = outputs[:, 1:, :].contiguous()
        target = tgt[:, 1:].contiguous()

        B, Tm1, V = output.shape
        loss = criterion(output.view(B * Tm1, V), target.view(B * Tm1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, pad_idx, device="cpu"):
    model.eval()
    epoch_loss = 0.0

    for src, src_lens, tgt, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
        output = outputs[:, 1:, :].contiguous()
        target = tgt[:, 1:].contiguous()

        B, Tm1, V = output.shape
        loss = criterion(output.view(B * Tm1, V), target.view(B * Tm1))

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


########################################
# Inference for accuracy
########################################



@torch.no_grad()
def transliterate_word(model, word, src_stoi, tgt_itos, tgt_stoi, max_len=30):
    model.eval()
    device = next(model.parameters()).device

    # Encode source word
    src_ids = [src_stoi[ch] for ch in word if ch in src_stoi]
    if not src_ids:
        return ""
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
    memory, src_key_padding_mask = model.encoder(src_tensor)  # (1, L, D)
    
    PAD_IDX = tgt_stoi[PAD_TOKEN]
    SOS_IDX = tgt_stoi[SOS_TOKEN]
    EOS_IDX = tgt_stoi[EOS_TOKEN]

    # Start tgt with <sos>
    tgt_ids = [SOS_IDX]
    for _ in range(max_len):
        tgt_input = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        logits = model.decoder(
            tgt_input,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (1, t, vocab)

        next_token_logits = logits[0, -1, :]  # last position
        next_id = next_token_logits.argmax(-1).item()

        if next_id == EOS_IDX or next_id == PAD_IDX:
            break

        tgt_ids.append(next_id)

    # Convert ids → string (skip <sos>)
    chars = [tgt_itos[idx] for idx in tgt_ids[1:]]
    return "".join(chars)


@torch.no_grad()
def compute_exact_match_accuracy(model, df, src_stoi, tgt_itos, sos_idx, eos_idx, tgt_stoi, pad_idx, device):
    model.eval()
    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        native = row["native"]
        gold = row["roman"]
        pred = transliterate_word(model, native, src_stoi, tgt_itos, sos_idx, eos_idx, tgt_stoi, pad_idx, device)
        if pred == gold:
            correct += 1

    return correct / total if total > 0 else 0.0


########################################
# Main
########################################

def main():
    parser = argparse.ArgumentParser(description="Train Hindi→Roman Transformer transliteration model.")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing train.tsv, val.tsv, test.tsv.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--enc_emb_dim", type=int, default=128)
    parser.add_argument("--dec_emb_dim", type=int, default=128)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./artifacts",
                        help="Directory to save model and vocab jsons.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    train_df = pd.read_csv(data_dir / "train.tsv", sep="\t")
    val_df = pd.read_csv(data_dir / "val.tsv", sep="\t")
    test_df = pd.read_csv(data_dir / "test.tsv", sep="\t")

    # Build vocab
    src_stoi, tgt_stoi, src_itos, tgt_itos, pad_idx, sos_idx, eos_idx = build_vocabs(train_df)

    # Save vocab jsons (for app.py / HF model)
    with open(save_dir / "src_stoi.json", "w", encoding="utf-8") as f:
        json.dump(src_stoi, f, ensure_ascii=False)
    with open(save_dir / "tgt_stoi.json", "w", encoding="utf-8") as f:
        json.dump(tgt_stoi, f, ensure_ascii=False)

    # Datasets & loaders
    train_dataset = TransliterationDataset(train_df, src_stoi, tgt_stoi, sos_idx, eos_idx)
    val_dataset = TransliterationDataset(val_df, src_stoi, tgt_stoi, sos_idx, eos_idx)
    test_dataset = TransliterationDataset(test_df, src_stoi, tgt_stoi, sos_idx, eos_idx)

    collate = lambda batch: collate_fn(batch, pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model
    input_dim = len(src_stoi)
    output_dim = len(tgt_stoi)

    encoder = TransformerEncoder(input_dim, args.enc_emb_dim, args.hid_dim,
                         num_layers=1, dropout=args.dropout, pad_idx=pad_idx)
    decoder = TransformerDecoder(output_dim, args.dec_emb_dim, args.hid_dim,
                         num_layers=1, dropout=args.dropout, pad_idx=pad_idx)

    model = Seq2SeqTransformer(encoder, decoder, pad_idx, sos_idx, eos_idx, device).to(device)

    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_model_path = save_dir / "best_hindi_roman_transformer.pt"

    for epoch in range(1, args.epochs + 1):
        # simple teacher forcing schedule
        tf_ratio = max(0.3, 0.7 - (epoch - 1) * 0.02)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     pad_idx, teacher_forcing_ratio=tf_ratio, device=device)
        val_loss = evaluate(model, val_loader, criterion, pad_idx, device=device)

        print(f"[Epoch {epoch:02d}] TF={tf_ratio:.2f} "
              f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  [INFO] Saved new best model to: {best_model_path}")

    # Evaluate exact-match accuracy using best model
    print("[INFO] Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Build tgt_itos list from tgt_stoi
    max_idx = max(tgt_stoi.values())
    tgt_itos_list = [""] * (max_idx + 1)
    for ch, idx in tgt_stoi.items():
        tgt_itos_list[idx] = ch

    val_acc = compute_exact_match_accuracy(model, val_df, src_stoi, tgt_itos_list,
                                           sos_idx, eos_idx, pad_idx, tgt_stoi, device)
    test_acc = compute_exact_match_accuracy(model, test_df, src_stoi, tgt_stoi, tgt_itos_list,
                                            sos_idx, eos_idx, pad_idx, device)

    print(f"[RESULT] Validation exact-match accuracy: {val_acc * 100:.2f}%")
    print(f"[RESULT] Test exact-match accuracy:       {test_acc * 100:.2f}%")

    print(f"[INFO] Vocab + model artifacts saved in: {save_dir}")


if __name__ == "__main__":
    main()
