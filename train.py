# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import math
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
from dataclasses import dataclass

# -----------------------------
# Step 1: Define helper functions
# -----------------------------
def get_batch(split):
    if split == 'train':
        data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('data/validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# -----------------------------
# Step 2: Define Model Architecture
# -----------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

# ... [Copy the rest of GPT model classes here from your notebook]
# For brevity, I’m skipping retyping full model code; use the same classes you already have: CausalSelfAttention, MLP, Block, GPTConfig, GPT

# -----------------------------
# Step 3: Training Configuration
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'

learning_rate = 1e-4
max_iters = 20000
warmup_steps = 1000
min_lr = 5e-4
eval_iters = 500
batch_size = 32
block_size = 128
gradient_accumulation_steps = 32

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type=='cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------
# Step 4: Initialize Model, Optimizer, Scheduler
# -----------------------------
config = GPTConfig(vocab_size=50257, block_size=block_size, n_layer=6, n_head=6, n_embd=384, dropout=0.1)
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.95), weight_decay=0.1, eps=1e-9)
scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters-warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))

# -----------------------------
# Step 5: Training Loop
# -----------------------------
best_val_loss = float('inf')
best_model_params_path = "saved_models/best_model_params.pt"
train_loss_list, validation_loss_list = [], []

for epoch in tqdm(range(max_iters)):
    if epoch % eval_iters == 0 and epoch != 0:
        losses = estimate_loss(model)
        print(f"Epoch {epoch}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        train_loss_list.append(losses['train'])
        validation_loss_list.append(losses['val'])
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), best_model_params_path)

    X, y = get_batch("train")
    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if ((epoch+1) % gradient_accumulation_steps == 0) or (epoch+1==max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()
