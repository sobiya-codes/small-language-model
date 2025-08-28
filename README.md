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
│
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
└── README.md


---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/small-language-model.git
cd small-language-model


Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows


Install dependencies:

pip install -r requirements.txt


Note: GPU with CUDA is recommended for faster training.

📝 How to Run
1. Prepare the dataset
python data_preparation.py


This will tokenize TinyStories dataset and create train.bin and validation.bin.

2. Train the model
python train.py


Model weights will be saved in saved_models/best_model_params.pt.

3. Generate text
python inference.py


You can edit inference.py to provide your own prompt and control output length.

🎯 Example Outputs

Prompt:
Once upon a time there was a pumpkin.

Generated Text:
Once upon a time there was a pumpkin. It was orange and round, and everyone loved it. The pumpkin rolled around in the garden and made friends with the little mice...

Prompt:
A little girl went to the woods

Generated Text:
A little girl went to the woods. She saw the birds singing and the flowers blooming. Suddenly, she found a hidden path that led to a magical treehouse...

🛠️ Technical Details

Model Architecture: GPT-like small transformer

Layers: 6

Heads: 6

Embedding Size: 384

Block size (context): 128 tokens

Dataset: TinyStories (HuggingFace)

Training: Mixed precision (bfloat16/float16)

Optimizer: AdamW with weight decay

Learning Rate Scheduler: Linear warmup + Cosine decay

👏 Credits

Dataset: TinyStories by Ronen Eldan

Model Inspiration & Training Code: Vizuara AI Labs

📜 License

This project is open-sourced for learning purposes.
Please give credit to Vizuara AI Labs if you use this implementation.


