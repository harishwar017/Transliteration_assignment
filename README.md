# Transliteration_assignment

# Hindi → Roman script Transliteration using a Character-Level Transformer encoder–decoder model

This project implements **lightweight character-level Transformer encoder–decoder model** (0.6 M) that transliterates **Hindi words into Roman (Latin) script**. I have also experimented with GRU based seq-seq model and have compared both.
The model is trained from scratch using the **Dakshina transliteration dataset** and deployed as a live demo using **HuggingFace Spaces**.

- Final model link:'https://huggingface.co/spaces/harishwar017/transliteration_transformer'
- Training dataset link:'https://huggingface.co/datasets/harishwar017/translit_training_data'
- Validation dataset link:'https://huggingface.co/datasets/harishwar017/translit_test_data'

---

## What is Transliteration?

**Transliteration** is the process of converting text from one script to another **without changing the language**.

Example:

| Hindi | Roman |
|-------|--------|
| नमस्कार | namaskar |
| घरों | gharon |
| सफाई | safai |

This is **not translation** (meaning is unchanged).

---

## Model Overview

- **Architecture:** Character-level Seq2Seq (GRU/Transformer Encoder–Decoder)
- **Input:** Hindi word (Devanagari script)
- **Output:** Romanized word (Latin script)
- **Training Type:** Fully supervised
- **Special Tokens:** `<pad>`, `<sos>`, `<eos>`

### Model Hyperparameters

| Parameter | Value |
|----------|--------|
| Encoder Embedding | 128 |
| Decoder Embedding | 128 |
| Hidden Size | 256 |
| GRU Layers | 1 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning Rate | 1e-3 |

---

## Dataset

The dataset is built using the **Dakshina Hindi Lexicon Splits**:

- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

Each row contains:


### Dataset Sizes

| Split | Samples |
|--------|----------|
| Train | 79,805 |
| Validation | 4,358 |
| Test | 4,502 |

- Only **clean word-level data** is used  
- Roman outputs are normalized to **lowercase a–z only**

---
## 📊 Evaluation Metrics

| Model | Time |
|--------|----------|
| GRU | 30 mins |
| Transformer | 3 mins % |

## 📊 Evaluation Metrics

- **Primary Metric:** Exact Match Word Accuracy  
- Accuracy is computed by comparing full predicted words with ground truth.

GRU Accuracy

| Split | Accuracy |
|--------|----------|
| Validation | 25.56 % |
| Test | 24.57 % |

Transformer Accuracy

| Split | Accuracy |
|--------|----------|
| Validation | 45.56 % |
| Test | 44.57 % |

---

## ⚡ Inference Latency

Average inference time per word:

GRU:

| Device | Latency |
|--------|----------|
| CPU | 3.329 ms / word |
| GPU | 2.921 ms / word |


Transformer:

| Device | Latency |
|--------|----------|
| CPU | 25.095 ms / word |
| GPU | 22.095 ms / word |


_(Measured using random Hindi words from the training set)_

---

## Live Demo (HuggingFace Spaces)

You can try sentence-level transliteration here:

🔗 **HuggingFace Space:**  
`https://huggingface.co/spaces/harishwar017/transliteration_transformer`

- Paste a **Hindi sentence**
- It is internally split into words
- Each word is transliterated
- Final Romanized sentence is returned with punctuation preserved

The trained model and vocabularies are hosted on HuggingFace:

🔗 **Other Experiments:**  
- GRU model: 'https://huggingface.co/harishwar017/hindi-roman-gru'

Files included:
- `best_hindi_roman_gru.pt`
- `src_stoi.json`
- `tgt_stoi.json`

---

### File Descriptions

| File Name | Purpose |
| :--- | :--- |
| `app.py` | Hosts the web interface (Gradio) for live model inference and demonstration, primarily for Hugging Face Spaces. |
| `train.py` | Contains the script for defining, compiling, and running the model training loop. |
| `data_prep.py` | Used to load, clean, tokenize, and process the raw dataset into a format usable for training. |
| `requirements.txt` | Lists all necessary Python dependencies (libraries and versions) for the project. |
| `src_stoi.json` | Stores the vocabulary mapping (string-to-index) for the source/input data tokens. |
| `tgt_stoi.json` | Stores the vocabulary mapping (string-to-index) for the target/output data tokens. |

