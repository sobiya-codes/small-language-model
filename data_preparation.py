# data_preparation.py
import os
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm.auto import tqdm

# Load dataset
ds = load_dataset("roneneldan/TinyStories")

# Initialize GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}

# Only process if train.bin doesn't exist
if not os.path.exists("data/train.bin"):
    os.makedirs("data", exist_ok=True)
    tokenized = ds.map(process, remove_columns=['text'], num_proc=8)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'data/{split}.bin'
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

print("Dataset preprocessing complete!")
