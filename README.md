# Small Language Model (SLM) from Scratch 👩‍💻

![Project Demo](assets/demo.gif)  

A **Small Language Model (SLM)** trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset to generate creative and coherent text.  
This project is implemented from scratch in **PyTorch**, inspired by the work of **Vizuara AI Labs**.  

---

## 🌟 Features

- Small-scale GPT-like architecture (~50-60M parameters)  
- Trained on TinyStories dataset (kid-friendly short stories)  
- Causal self-attention for next-token prediction  
- Mixed precision training for faster computation  
- Generate text given a prompt with controllable length and randomness  

---

## 📂 Project Structure

small-language-model/
├── assets/
│ └── demo.gif # Project demo GIF
├── data/
│ ├── train.bin
│ └── validation.bin
├── saved_models/
│ └── best_model_params.pt # Trained model weights
├── data_preparation.py # Tokenize dataset and create .bin files
├── train.py # Model training code
├── inference.py # Run inference / generate text
├── requirements.txt # Python dependencies
└── README.md # Project documentation
