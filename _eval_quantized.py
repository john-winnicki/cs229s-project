"""
Evaluate next-token-prediction perplexity of pre-trained model
"""

import gc
from model import GPTConfig, GPT
from model_quantized import GPT_QConfig, GPT_Q
from contextlib import nullcontext
import numpy as np
import os
import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------------------------------------------------------
init_from = 'gpt2' # a gpt2 variant (e.g. 'gpt2-xl')
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
dataset = 'shakespeare' # 'shakespeare' or 'wikitext'
block_size = 1024
# -----------------------------------------------------------------------------

# Setup environment
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Load dataset
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Measure perplexity on dataset
# https://huggingface.co/docs/transformers/perplexity
# Table 3, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
def measure_perplexity(model, data, batch_size):
    nll_weights = []
    nlls = []
    for i in tqdm(range(0, len(data), block_size * batch_size)):
        j = min(i + block_size * batch_size, len(data))
        ix = torch.arange(i, j, block_size)
        x = []
        for k in ix:
            x.append(torch.from_numpy((data[k:k+block_size]).astype(np.int64)))
        x = torch.stack([F.pad(y, (0, block_size - len(y)), value=-1) for y in x])
        nll_weights.append((x != -1).sum().item() / len(data))
        if device_type == 'cuda':
            # pin array x which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        with torch.no_grad():
            with ctx:
                y = x[:, 1:].clone()
                x[x == -1] = 0
                logits, loss = model(x[:, :-1], y)
                nlls.append(loss)
                # # https://github.com/huggingface/transformers/blob/df5c5c62ae253055336f5bb0828ca8e3e15ab6bd/src/transformers/models/gpt2/modeling_gpt2.py#L1099
                # y = x.clone()
                # x[x == -1] = 0
                # logits, _ = model(x, y)
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = y[..., 1:].contiguous()
                # loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
                # nlls.append(loss)
    nlls = [nll_weights[i] * nlls[i] for i in range(len(nlls))]
    return torch.exp(torch.stack(nlls).sum()).item()

# Load pre-trained model
model_pt = GPT.from_pretrained(init_from, dict(dropout=0.0))
model_pt.eval()
torch.cuda.reset_peak_memory_stats(device=device)
model_pt.to(device)
print(f"GPU memory allocated after moving pre-trained model to device {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model_pt = torch.compile(model_pt) # requires PyTorch 2.0 (optional)

torch.cuda.reset_peak_memory_stats(device=device)
ppl_pt_bs4 = measure_perplexity(model_pt, val_data, batch_size=4)
print(f"GPT-2 perplexity on {dataset}/val.bin, batch_size={4}: {ppl_pt_bs4:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
ppl_pt_bs12 = measure_perplexity(model_pt, val_data, batch_size=12)
print(f"GPT-2 perplexity on {dataset}/val.bin, batch_size={12}: {ppl_pt_bs12:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
del model_pt
gc.collect()
torch.cuda.empty_cache()

# Load pre-trained model and quantize
model_dq = GPT_Q.from_pretrained(init_from, dict(dropout=0.0))
model_dq.quantize_all_parameters()
torch.cuda.reset_peak_memory_stats(device=device)
model_dq.to(device)
print(f"GPU memory allocated after moving quantized model to device {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model_dq = torch.compile(model_dq) # requires PyTorch 2.0 (optional)

torch.cuda.reset_peak_memory_stats(device=device)
ppl_dq_bs4 = measure_perplexity(model_dq, val_data, batch_size=4)
print(f"GPT-2 dynamic quantized perplexity on {dataset}/val.bin, batch_size={4}: {ppl_dq_bs4:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
ppl_dq_bs12 = measure_perplexity(model_dq, val_data, batch_size=12)
print(f"GPT-2 dynamic quantized perplexity on {dataset}/val.bin, batch_size={12}: {ppl_dq_bs12:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
del model_dq
gc.collect()
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# To minimize gpu memory usage, would it help to shuffle tensors back and forth 
# from GPU?
# -----------------------------------------------------------------------------