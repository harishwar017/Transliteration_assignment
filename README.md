# Transliteration_assignment

# Hindi → Roman Transliteration using Character-Level GRU

This project implements a **lightweight character-level GRU encoder–decoder model** that transliterates **Hindi words into Roman (Latin) script**.  
The model is trained from scratch using the **Dakshina transliteration dataset** and deployed as a live demo using **HuggingFace Spaces**.

---

## 🔹 What is Transliteration?

**Transliteration** is the process of converting text from one script to another **without changing the language**.

Example:

| Hindi | Roman |
|-------|--------|
| नमस्कार | namaskar |
| घरों | gharon |
| सफाई | safai |

This is **not translation** (meaning is unchanged).

---

## ✅ Model Overview

- **Architecture:** Character-level Seq2Seq (GRU Encoder–Decoder)
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

## 📂 Dataset

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

✅ Only **clean word-level data** is used  
✅ Roman outputs are normalized to **lowercase a–z only**

---

## 📊 Evaluation Metrics

- **Primary Metric:** Exact Match Word Accuracy  
- Accuracy is computed by comparing full predicted words with ground truth.

| Split | Accuracy |
|--------|----------|
| Validation | 25.56 % |
| Test | 24.57 % |

_(Fill these numbers after your final evaluation run)_

---

## ⚡ Inference Latency

Average inference time per word:

| Device | Latency |
|--------|----------|
| CPU | 3.329 ms / word |
| GPU | 2.921 ms / word |

_(Measured using random Hindi words from the training set)_

---

## 🌐 Live Demo (HuggingFace Spaces)

You can try sentence-level transliteration here:

🔗 **HuggingFace Space:**  
`https://huggingface.co/spaces/harishwar017/YOUR_SPACE_NAME`

- Paste a **Hindi sentence**
- It is internally split into words
- Each word is transliterated
- Final Romanized sentence is returned with punctuation preserved

---

## 🤗 Pretrained Model

The trained model and vocabularies are hosted on HuggingFace:

🔗 **HuggingFace Model:**  
`https://huggingface.co/harishwar017/hindi-roman-gru`

Files included:
- `best_hindi_roman_gru.pt`
- `src_stoi.json`
- `tgt_stoi.json`

---

## 🛠 Project Structure

├── app.py # Gradio app for HuggingFace Space
├── train.py # Model training script
├── data_prep.py # Dataset processing script
├── requirements.txt
├── src_stoi.json
├── tgt_stoi.json
└── README.md

