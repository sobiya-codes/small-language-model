# 🧠 Small Language Model (SLM) from Scratch

![Project GIF](path_to_your_gif.gif)  
*An example generation from the SLM model.*

---

## 🔹 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
- [License](#license)

---

## 🔹 Project Overview
This project implements a **GPT-style transformer** from scratch, trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).  

The goal is to create a **creative and coherent text generator** using a lightweight model (~50M parameters).  

It demonstrates:
- Tokenization with `tiktoken`
- Efficient dataset preprocessing
- Transformer blocks (Multi-Head Attention, MLP, LayerNorm, Residual connections)
- Mixed-precision training with gradient clipping
- Text generation with temperature and top-k sampling

---

## 🔹 Features
- ✅ Tokenization & preprocessing using GPT-2 tokenizer  
- ✅ Custom GPT-style architecture with self-attention blocks  
- ✅ AdamW optimizer with cosine learning rate scheduler  
- ✅ Mixed precision training for speed and memory efficiency  
- ✅ Gradient accumulation & clipping for stable training  
- ✅ Text generation using configurable prompt, temperature, and top-k  

---

## 🔹 Dataset
- **Name:** TinyStories  
- **Source:** [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)  
- **Description:** Synthetic short stories for 3-4 year olds  
- **Usage:** Preprocessed into `train.bin` and `val.bin` for efficient memory usage

---

## 🔹 Architecture
**Model:** GPT-style transformer (~50M parameters)  

**Components:**
- **Embedding Layer:** Token + positional embeddings  
- **Transformer Blocks:** Multi-head causal self-attention + MLP + residual + LayerNorm  
- **Output Layer:** Linear layer projecting hidden states to vocabulary  

**Hyperparameters (example):**
- Layers: 6  
- Embedding dim: 384  
- Heads: 6  
- Context window: 128  
- Dropout: 0.1  

---

## 🔹 Setup & Installation
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/slm-project.git
cd slm-project

# Create virtual environment
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
