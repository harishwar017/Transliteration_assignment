#!/usr/bin/env python
"""
Evaluate Hindi → Roman transliteration model.

Assumes:
- Char-level GRU encoder-decoder model.
- Saved model state dict at --model_path (e.g. best_hindi_roman_gru.pt).
- src_stoi.json and tgt_stoi.json vocab files.
- data/{train,val,test}.tsv with columns: native, roman.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


# ========= Model definitions (must match training) =========

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


class EncoderGRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (B, src_len)
        embedded = self.dropout(self.embedding(src))          # (B, src_len, emb_dim)
        outputs, hidden = self.gru(embedded)                  # outputs: (B, src_len, H)
        return outputs, hidden


class DecoderGRU(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input: (B,)
        input = input.unsqueeze(1)  # (B, 1)
        embedded = self.dropout(self.embedding(input))        # (B, 1, emb_dim)
        output, hidden = self.gru(embedded, hidden)           # output: (B, 1, H)
        output = output.squeeze(1)                            # (B, H)
        logits = self.fc_out(output)                          # (B, vocab_size)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use transliterate_word for inference only.")


# ========= Inference helpers =========

@torch.no_grad()
def transliterate_word(model, word: str, src_stoi, tgt_itos, sos_idx, eos_idx, pad_idx, device, max_len: int = 30):
    """
    Greedy decoding for a single Hindi word → Roman string.
    """
    # Encode Hindi word as indices
    src_ids = [src_stoi[ch] for ch in word if ch in src_stoi]
    if not src_ids:
        return ""

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long)

    # Encode
    _, hidden = model.encoder(src_tensor)

    # Decode
    input_token = torch.tensor([sos_idx], dtype=torch.long, device=device)
    decoded = []

    for _ in range(max_len):
        logits, hidden = model.decoder(input_token, hidden)
        top1 = logits.argmax(1)
        idx = top1.item()

        if idx == eos_idx or idx == pad_idx:
            break

        decoded.append(idx)
        input_token = top1

    chars = []
    for idx in decoded:
        chars.append(tgt_itos[idx])
    return "".join(chars)


@torch.no_grad()
def compute_exact_match_accuracy(model, df, src_stoi, tgt_itos, sos_idx, eos_idx, pad_idx, device, max_len=30):
    """
    Exact-match word-level accuracy:
    fraction of words where predicted roman == gold roman.
    """
    model.eval()
    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        native = row["native"]
        gold = str(row["roman"])
        pred = transliterate_word(model, native, src_stoi, tgt_itos, sos_idx, eos_idx, pad_idx, device, max_len=max_len)
        if pred == gold:
            correct += 1

    return correct / total if total > 0 else 0.0


# ========= Main =========

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hindi→Roman GRU transliteration model.")
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing train.tsv / val.tsv / test.tsv (with 'native' and 'roman' columns)."
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Which split to evaluate on."
    )
    parser.add_argument(
        "--model_path", type=str, default="best_hindi_roman_gru.pt",
        help="Path to saved model state_dict (.pt)."
    )
    parser.add_argument(
        "--vocab_dir", type=str, default=".",
        help="Directory containing src_stoi.json and tgt_stoi.json."
    )
    parser.add_argument(
        "--max_len", type=int, default=30,
        help="Maximum decoding length in characters."
    )
    parser.add_argument(
        "--enc_emb_dim", type=int, default=128,
        help="Encoder embedding size (must match training)."
    )
    parser.add_argument(
        "--dec_emb_dim", type=int, default=128,
        help="Decoder embedding size (must match training)."
    )
    parser.add_argument(
        "--hid_dim", type=int, default=256,
        help="Hidden size (must match training)."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout used in training (not critical for eval, but keep consistent)."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_dir = Path(args.data_dir)
    vocab_dir = Path(args.vocab_dir)
    split_path = data_dir / f"{args.split}.tsv"

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    print(f"[INFO] Loading {args.split} split from: {split_path}")
    df = pd.read_csv(split_path, sep="\t")

    # Load vocab
    src_stoi_path = vocab_dir / "src_stoi.json"
    tgt_stoi_path = vocab_dir / "tgt_stoi.json"

    if not src_stoi_path.exists() or not tgt_stoi_path.exists():
        raise FileNotFoundError(f"Could not find src_stoi.json or tgt_stoi.json in {vocab_dir}")

    with open(src_stoi_path, "r", encoding="utf-8") as f:
        src_stoi = json.load(f)

    with open(tgt_stoi_path, "r", encoding="utf-8") as f:
        tgt_stoi = json.load(f)

    # Build tgt_itos list (index to character)
    max_idx = max(tgt_stoi.values())
    tgt_itos = [""] * (max_idx + 1)
    for ch, idx in tgt_stoi.items():
        tgt_itos[idx] = ch

    pad_idx = tgt_stoi[PAD_TOKEN]
    sos_idx = tgt_stoi[SOS_TOKEN]
    eos_idx = tgt_stoi[EOS_TOKEN]

    input_dim = len(src_stoi)
    output_dim = len(tgt_stoi)

    print(f"[INFO] Source vocab size: {input_dim}")
    print(f"[INFO] Target vocab size: {output_dim}")

    # Build model (must match training hyperparams)
    encoder = EncoderGRU(
        input_dim=input_dim,
        emb_dim=args.enc_emb_dim,
        hid_dim=args.hid_dim,
        num_layers=1,
        dropout=args.dropout,
        pad_idx=pad_idx,
    )
    decoder = DecoderGRU(
        output_dim=output_dim,
        emb_dim=args.dec_emb_dim,
        hid_dim=args.hid_dim,
        num_layers=1,
        dropout=args.dropout,
        pad_idx=pad_idx,
    )
    model = Seq2Seq(encoder, decoder, pad_idx, sos_idx, eos_idx, device).to(device)

    # Load state dict
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[INFO] Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Compute accuracy
    print("[INFO] Computing exact-match accuracy...")
    acc = compute_exact_match_accuracy(
        model=model,
        df=df,
        src_stoi=src_stoi,
        tgt_itos=tgt_itos,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        device=device,
        max_len=args.max_len,
    )

    print(f"[RESULT] {args.split} exact-match accuracy: {acc * 100:.2f}%")

    # Show a few qualitative examples
    print("\n[EXAMPLES]")
    for i in range(5):
        if i >= len(df):
            break
        row = df.iloc[i]
        native = row["native"]
        gold = str(row["roman"])
        pred = transliterate_word(model, native, src_stoi, tgt_itos, sos_idx, eos_idx, pad_idx, device, max_len=args.max_len)
        print(f"{i+1}. {native} → pred: {pred} | gold: {gold}")


if __name__ == "__main__":
    main()


# # Test split
# python evaluate.py \
#   --data_dir data \
#   --split test \
#   --model_path best_hindi_roman_gru.pt \
#   --vocab_dir . 

# # Or validation split
# python evaluate.py \
#   --data_dir data \
#   --split val \
#   --model_path best_hindi_roman_gru.pt \
#   --vocab_dir .

# --enc_emb_dim 128 --dec_emb_dim 128 --hid_dim 256 --dropout 0.2
