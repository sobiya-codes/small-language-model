# inference.py
import torch
import tiktoken
from train import GPT, GPTConfig  # import the model and config from train.py

# -----------------------------
# Step 1: Load tokenizer
# -----------------------------
enc = tiktoken.get_encoding("gpt2")

# -----------------------------
# Step 2: Load trained model
# -----------------------------
config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained weights
model.load_state_dict(torch.load("saved_models/best_model_params.pt", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Step 3: Function to generate text
# -----------------------------
def generate_text(prompt, max_new_tokens=200, temperature=1.0, top_k=None):
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)
    with torch.no_grad():
        idx = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(idx.squeeze().tolist())

# -----------------------------
# Step 4: Example usage
# -----------------------------
if __name__ == "__main__":
    prompts = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods"
    ]
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        output = generate_text(prompt, max_new_tokens=200)
        print(f"Generated: {output}\n{'-'*80}\n")
