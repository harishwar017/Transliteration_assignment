#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path("dakshina_dataset_v1.0/hi")

LEXICON_DIR = BASE_DIR / "lexicons"
ROMANIZED_DIR = BASE_DIR / "romanized"

def is_pure_punct(token: str) -> bool:
    """
    True if token is only punctuation / whitespace (no letters or digits).
    Works for any script.
    """
    if not isinstance(token, str):
        return True
    token = token.strip()
    if token == "":
        return True
    return all(not ch.isalnum() for ch in token)


def clean_roman(s: str) -> str:
    """
    Normalize roman strings:
    - lowercase
    - keep only a-z
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    return "".join(ch for ch in s if "a" <= ch <= "z")

import re

# Matches only Devanagari (Hindi) characters + spaces
HINDI_REGEX = re.compile(r'^[\u0900-\u097F\s]+$')

# Matches only Roman letters + spaces
ROMAN_REGEX = re.compile(r'^[A-Za-z\s]+$')

def is_valid_hindi(text: str) -> bool:
    return isinstance(text, str) and bool(HINDI_REGEX.match(text))

def is_valid_roman(text: str) -> bool:
    return isinstance(text, str) and bool(ROMAN_REGEX.match(text))


def load_romanized_nopunct() -> pd.DataFrame:
    path = ROMANIZED_DIR / f"hi.romanized.rejoined.aligned.cased_nopunct.tsv"
    
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["native", "roman", "freq"],
        encoding="utf-8"
    )
    
    df = df.dropna(subset=["native", "roman"])
    df = df[~df["native"].apply(is_pure_punct)].copy()
    df = df[
    df["native"].apply(is_valid_hindi) &
    df["roman"].apply(is_valid_roman)
].copy()
    
    return df


def load_lexicon_split(lexicon_dir: Path, split: str) -> pd.DataFrame:
    """
    Load Dakshina Hindi lexicon TSV:
    hi.translit.sampled.{split}.tsv

    Format: native \t roman \t freq
    We drop freq and pure punctuation rows.
    """
    path = lexicon_dir / f"hi.translit.sampled.{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")

    df = pd.read_csv(path, sep="\t", header=None, names=["native", "roman", "freq"])
    df = df.dropna(subset=["native", "roman"])
    df = df[~df["native"].apply(is_pure_punct)].copy()

    return df[["native", "roman"]]


def build_splits(hi_root: Path):
    lexicon_dir = hi_root / "lexicons"
    if not lexicon_dir.exists():
        raise FileNotFoundError(f"Lexicon directory not found: {lexicon_dir}")

    print(f"[INFO] Loading lexicon splits from: {lexicon_dir}")

    lex_train = load_lexicon_split(lexicon_dir, "train")
    lex_dev = load_lexicon_split(lexicon_dir, "dev")
    lex_test = load_lexicon_split(lexicon_dir, "test")
    
    roman_train = load_romanized_nopunct()
    
    train_df = pd.concat(
    [lex_train[["native", "roman"]],
     roman_train[["native", "roman"]]
     ],
    ignore_index=True
)

    print(f"[INFO] Raw lexicon sizes - train: {len(lex_train)}, "
          f"dev: {len(lex_dev)}, test: {len(lex_test)}")

    # Clean roman side: lowercase a-z only
    for name, df in [("train", train_df), ("dev", lex_dev), ("test", lex_test)]:
        df["roman_clean"] = df["roman"].astype(str).apply(clean_roman)
        before = len(df)
        df.drop(df[df["roman_clean"] == ""].index, inplace=True)
        after = len(df)
        print(f"[INFO] {name}: dropped {before - after} rows with empty roman after cleaning")

    train_df = train_df[["native", "roman_clean"]].rename(columns={"roman_clean": "roman"})
    val_df = lex_dev[["native", "roman_clean"]].rename(columns={"roman_clean": "roman"})
    test_df = lex_test[["native", "roman_clean"]].rename(columns={"roman_clean": "roman"})

    # Drop duplicates
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    val_df = val_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    print(f"[INFO] Final sizes - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Prepare Hindi→Roman word-level dataset from Dakshina.")
    parser.add_argument(
        "--hi_root",
        type=str,
        required=True,
        help="Path to dakshina_dataset_v1.0/hi directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data",
        help="Output directory to save processed train/val/test TSVs.",
    )

    args = parser.parse_args()
    hi_root = Path(args.hi_root)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = build_splits(hi_root)

    train_path = out_dir / "train.tsv"
    val_path = out_dir / "val.tsv"
    test_path = out_dir / "test.tsv"

    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)

    print(f"[INFO] Saved train to: {train_path}")
    print(f"[INFO] Saved val   to: {val_path}")
    print(f"[INFO] Saved test  to: {test_path}")


if __name__ == "__main__":
    main()


# python data_prep.py \
#   --hi_root /content/dakshina_dataset_v1.0/hi \
#   --out_dir ./data

# will create
# ./data/train.tsv
# ./data/val.tsv
# ./data/test.tsv
