import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

import torch.nn as nn












# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 5
log_interval = 1
eval_iters = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'cs229s'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'shakespeare'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768

# # model
# n_layer = 6
# n_head = 6
# n_embd = 384

dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
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

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

#################################################################
#################################################################
# JOHN 12/09/2023

def l1_unstructured_prune(module, name, amount):
    """Apply L1 unstructured pruning to a specified parameter of a module,
    following PyTorch's pruning convention.

    Args:
        module (nn.Module): The module containing the parameter to prune.
        name (str): The name of the parameter within the module to prune.
        amount (float): The fraction of connections to prune (between 0 and 1).

    Raises:
        ValueError: If the amount is not between 0 and 1.
    """
    # Ensure amount is a valid fraction
    if not 0 <= amount <= 1:
        raise ValueError("Pruning amount must be a fraction between 0 and 1.")

    # Retrieve the tensor to prune
    tensor = getattr(module, name)

    # Compute the number of parameters to prune
    num_params_to_prune = round(amount * tensor.nelement())

    # Compute the mask
    if num_params_to_prune != 0:
        l1_norm = torch.abs(tensor.view(-1))
        threshold = torch.topk(l1_norm, num_params_to_prune, largest=False).values[-1]
        mask = torch.gt(l1_norm, threshold).float().view_as(tensor)
    else:
        mask = torch.ones_like(tensor)

    # Save the original tensor and the mask
    module.register_parameter(name + "_orig", torch.nn.Parameter(tensor.detach()))
    module.register_buffer(name + "_mask", mask)

    # Define the hook for applying the mask
    def apply_mask_hook(module, inputs):
        orig_tensor = getattr(module, name + "_orig")
        mask = getattr(module, name + "_mask")
        pruned_tensor = orig_tensor * mask
        # Set the pruned tensor as a parameter temporarily for the forward pass
        setattr(module, name, torch.nn.Parameter(pruned_tensor))


    # Attach the hook to the module
    hook = module.register_forward_pre_hook(apply_mask_hook)

    return hook



# def l2_structured_pruning(model, current_layer_name, previous_layer_name, amount):
#     """
#     Apply L2 structured pruning to a specified linear layer of a model, 
#     and adjust the dimensions of the previous linear layer accordingly.
#     """
#     print("\n\n\n\n")
#     print(getattr(model, current_layer_name).in_features, getattr(model, current_layer_name).out_features)
#     # print(getattr(model, previous_layer_name).in_features, getattr(model, previous_layer_name).out_features)
#     current_layer = getattr(model, current_layer_name)
#     # previous_layer = getattr(model, previous_layer_name)

#     # if not (isinstance(current_layer, nn.Linear) and isinstance(previous_layer, nn.Linear)):
#         # raise TypeError("Both layers must be of type nn.Linear.")

#     # Compute L2 norm for each row in the weight matrix of the current layer
#     l2_norm = torch.norm(current_layer.weight.data, p=2, dim=1)
#     num_rows_to_keep = int((1 - amount) * l2_norm.size(0))
#     rows_to_keep = torch.topk(l2_norm, num_rows_to_keep, largest=True).indices

#     # print("aw09da09wdopjiawda098a6a5w5wa56wa", previous_layer.weight.data.shape, previous_layer.in_features)

#     # Create new pruned layer for current_layer
#     new_current_layer = nn.Linear(current_layer.in_features, num_rows_to_keep, bias=current_layer.bias is not None) #(384, 372)
#     new_current_layer.weight.data = current_layer.weight.data[rows_to_keep, :]
#     if current_layer.bias is not None:
#         new_current_layer.bias.data = current_layer.bias.data[rows_to_keep]

#     # # Adjust previous_layer to match the new dimension of current_layer
#     # new_previous_layer = nn.Linear(num_rows_to_keep, previous_layer.out_features, bias=previous_layer.bias is not None)
#     # new_previous_layer.weight.data = previous_layer.weight.data[:, rows_to_keep]

#     print("aw0d9aiud09aiwk", current_layer.weight.shape)
#     # print("aw0d9aiud09aiwk", previous_layer.weight.shape)

#     print("aw0d9aiud09aiwk", new_current_layer.weight.shape)
#     # print("aw0d9aiud09aiwk", new_previous_layer.weight.shape)

#     # Replace the old layers with the new pruned layers in the model
#     setattr(model, current_layer_name, new_current_layer)
#     # setattr(model, previous_layer_name, new_previous_layer)
#     print(getattr(model, current_layer_name).weight.shape)
#     # print(getattr(model, previous_layer_name).weight.shape)
    
#     print(getattr(model, current_layer_name).in_features, getattr(model, current_layer_name).out_features)
#     # print(getattr(model, previous_layer_name).in_features, getattr(model, previous_layer_name).out_features)

def l2_structured_attn_pruning(model, name, name2, amount, n_head):
    """
    Apply L2 structured pruning to a specified linear layer of a model, 
    and adjust the dimensions of the previous linear layer accordingly.
    """

    c_attn_layer = getattr(model, name)
    # print("CPROJ ORIGINAL SHAPE: ", getattr(model, "c_proj").weight.data.shape)
    print("ORIGINAL: ", getattr(model, name).weight.data.shape)

    qkv_weights = c_attn_layer.weight.data
    # print(qkv_weights.shape)
    n_embd = qkv_weights.shape[0]//3
    # print(n_embd)
    q, k, v = qkv_weights.split(n_embd, dim=0)
    # print(q.shape, k.shape, v.shape)

    ql2_norm = torch.norm(q, p=2, dim=1)
    qnum_rows_to_keep = int((1 - amount) * ql2_norm.size(0))
    qnum_rows_to_keep -= qnum_rows_to_keep % n_head
    qrows_to_keep = torch.topk(ql2_norm, qnum_rows_to_keep, largest=True).indices

    kl2_norm = torch.norm(k, p=2, dim=1)
    knum_rows_to_keep = int((1 - amount) * kl2_norm.size(0))
    knum_rows_to_keep -= knum_rows_to_keep % n_head
    krows_to_keep = torch.topk(kl2_norm, knum_rows_to_keep, largest=True).indices

    vl2_norm = torch.norm(v, p=2, dim=1)
    vnum_rows_to_keep = int((1 - amount) * vl2_norm.size(0))
    # print(vnum_rows_to_keep)
    vnum_rows_to_keep -= vnum_rows_to_keep % n_head
    # print("NEW: ", n_head, vnum_rows_to_keep)
    vrows_to_keep = torch.topk(vl2_norm, vnum_rows_to_keep, largest=True).indices

    # Create new pruned layer for current_layer
    # print("BIAS: ", c_attn_layer.bias)
    new_current_layer = nn.Linear(c_attn_layer.in_features, qnum_rows_to_keep+knum_rows_to_keep+vnum_rows_to_keep, bias=c_attn_layer.bias is not None)
    new_current_layer.weight.data = torch.cat([q[qrows_to_keep, :] , k[krows_to_keep, :] , v[vrows_to_keep, :] ], dim=0)

    setattr(model, name, new_current_layer)
    
    print("PRUNED: ", getattr(model, name).weight.data.shape)

    c_proj_layer = getattr(model, name2)

    # Adjust c_proj layer to match the new dimension of the pruned V part
    # Since c_proj is a Linear layer with shape [n_embd, n_embd], and only the input dimension needs to change
    new_c_proj_layer = nn.Linear(vnum_rows_to_keep, c_proj_layer.out_features, bias=c_proj_layer.bias is not None)

    # Copy the relevant weights and bias from the original c_proj layer to the new pruned layer
    # We take the columns corresponding to the kept rows in V
    new_c_proj_layer.weight.data = c_proj_layer.weight.data[:, vrows_to_keep]
    if c_proj_layer.bias is not None:
        new_c_proj_layer.bias.data = c_proj_layer.bias.data[vrows_to_keep]

    # Replace the old c_proj layer with the new pruned layer in the model
    setattr(model, name2, new_c_proj_layer)
    
    # print(getattr(model, name2).in_features, getattr(model, name2).out_features, getattr(model, name2).weight.data.shape)





# def find_linear_pairs(module):
#     """
#     Find pairs of consecutive linear layers in a given module.

#     Args:
#         module (torch.nn.Module): The module (or block) to search through.

#     Returns:
#         List of tuples: Each tuple contains (previous_layer_name, current_layer_name).
#     """
#     linear_pairs = []
#     previous_module = None
#     previous_name = None

#     # Iterate through all submodules and their names within the given module
#     for name, submodule in module.named_modules():
#         # Check if both the current and previous submodules are linear layers
#         if isinstance(previous_module, nn.Linear) and isinstance(submodule, nn.Linear):
#             linear_pairs.append((previous_name, name))
        
#         # Update the previous module and name
#         previous_module = submodule
#         previous_name = name

#     return linear_pairs

def apply_unstructured_pruning_to_causal_self_attention(module, pruning_percentage=0.2):
    l1_unstructured_prune(module.c_attn, name='weight', amount=pruning_percentage)

def apply_structured_pruning_to_causal_self_attention(block, pruning_percentage=0.2):
    l2_structured_attn_pruning(block, 'c_attn', 'c_proj', pruning_percentage, n_head)

def prune_gpt_model(model, pruning_percentage=0.2):
    # Iterate over all blocks in the transformer and apply pruning to the CausalSelfAttention layers
    for block in model.transformer.h:
        apply_unstructured_pruning_to_causal_self_attention(block.attn, pruning_percentage)
        # apply_structured_pruning_to_causal_self_attention(block.attn, pruning_percentage)

def calculate_reduction_percentage(current_percentage, target_percentage):
    """
    Calculate the percentage reduction needed to go from the current percentage
    of the original size to the target percentage of the original size.

    :param current_percentage: Current size as a percentage of the original size.
    :param target_percentage: Desired size as a percentage of the original size.
    :return: Reduction percentage required.
    """
    # Calculate the reduction factor
    reduction_factor = target_percentage / current_percentage

    # Convert the reduction factor to a percentage reduction
    reduction_percentage = (1 - reduction_factor) * 100

    return reduction_percentage

def calculate_percentage_to_reduce_each_step(start_percentage, step_reduction, steps):
    """
    Calculate the percentage to reduce at each step to decrease by a fixed percentage each time.

    :param start_percentage: Starting percentage of the original size.
    :param step_reduction: Fixed percentage reduction to achieve at each step.
    :param steps: Number of steps for which the reduction is to be calculated.
    :return: List of percentages to reduce at each step.
    """
    percentages_to_reduce = []
    current_percentage = start_percentage

    for _ in range(steps):
        # Calculate the target percentage for the next step
        target_percentage = current_percentage - step_reduction
        # Calculate the percentage to reduce at this step
        reduction_percentage = calculate_reduction_percentage(current_percentage, target_percentage)
        percentages_to_reduce.append(reduction_percentage)
        # Update the current percentage
        current_percentage = target_percentage

    return percentages_to_reduce

# Calculate the percentage to reduce at each step to decrease by 10% each time
percentages_to_reduce_each_step = calculate_percentage_to_reduce_each_step(100, 10, 10)  # 10 steps

#################################################################
#################################################################

#######################################################
#######################################################
#JOHN INFERENCE FUNCTION: 12/11/2023
def inferMe(model):
    # -----------------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------------
    init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out' # ignored if init_from is not 'resume'
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 1 # number of samples to draw
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        # print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        # print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)

    # function to measure inference latency
    def measure_inference_latency(batch_size):
        x = torch.tensor(start_ids * batch_size, dtype=torch.long, device=device).reshape(batch_size, -1)
        with torch.no_grad():
            with ctx:
                tokens_decoded = 0
                inference_start = time.time()
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    tokens_decoded += y.shape[0] * (y.shape[1] - 1)
                inference_finish = time.time()
        inference_total_seconds = inference_finish - inference_start
        inference_tokens_per_second = tokens_decoded / inference_total_seconds
        print(f"Inference latency, batch size {batch_size}: {inference_tokens_per_second:.4f} tokens/second")

    # print("\nMeasuring inference latency\n")

    batch_size = 1
    measure_inference_latency(1)
    batch_size = 12
    measure_inference_latency(12)
#############################################################
#############################################################

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

print("\nMeasuring training quality\n")

steps_since_prune = 0
prune_amount = 0

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    ############################################################
    ############################################################
    # if iter_num>=100 and iter_num%100==0: 
    #     model.eval()
    #     inferMe(model)
    #     model.train()
    ############################################################
    ############################################################

    ############################################################
    ############################################################
    #JOHN 12/09/2023: PRUNING 
    #JOHN 12/11/2023

    #STRUCTURED
    # if iter_num>=100 and iter_num%100==0:
    #     prune_gpt_model(model, pruning_percentage=percentages_to_reduce_each_step[int((iter_num//100) - 1)]/100)
    #     print("Iteration", iter_num, "PRUNED")

    # #UNSTRUCTURED
    # if iter_num>=100 and iter_num%100==0:
    #     prune_gpt_model(model, pruning_percentage=iter_num/1000)
    #     print("Iteration", iter_num, "PRUNED",iter_num/1000, "Percent of Graph")

    ############################################################
    ############################################################

    # determine and set the learning rate for this iteration

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    #JOHN 12/11/2023: COMMENTED OUT THIS PART SINCE UNNECESSARY
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
        #######################################################
        #######################################################

        #THIS NEEDS TO BE ACTIVATED WHEN RUNNING UNSTRUCTURED PRUNING

        # JOHN 12/09/2023
        # Apply mask to gradients for each parameter. This ensures zero 
        # values at all gradient elements where it should be zero by pruning
        for name, param in model.named_parameters():
            mask_name = name + '_mask'
            if mask_name in model.state_dict() and param.grad is not None:
                mask = model.state_dict()[mask_name]
                param.grad.data *= mask
        #######################################################
        #######################################################

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        #JOHN CHANGE 12/11/2023
        if iter_num>=50 and lossf<=3.2 and steps_since_prune>=25:
            prune_amount = prune_amount + 0.1
            prune_gpt_model(model, pruning_percentage=prune_amount)
            print("Iteration", iter_num, "PRUNED",prune_amount*100, "Percent of Graph")
            steps_since_prune = 0
        steps_since_prune += 1


    iter_num += 1
    local_iter_num += 1

    # termination conditions
    #JOHN CHANGE 12/11/2023
    # if iter_num >= 1000:
        # break
    if prune_amount==0.9:
        break










# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)

# function to measure inference latency
def measure_inference_latency(batch_size):
    x = torch.tensor(start_ids * batch_size, dtype=torch.long, device=device).reshape(batch_size, -1)
    with torch.no_grad():
        with ctx:
            tokens_decoded = 0
            inference_start = time.time()
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                tokens_decoded += y.shape[0] * (y.shape[1] - 1)
            inference_finish = time.time()
    inference_total_seconds = inference_finish - inference_start
    inference_tokens_per_second = tokens_decoded / inference_total_seconds
    print(f"Inference latency, batch size {batch_size}: {inference_tokens_per_second:.4f} tokens/second")

print("\nMeasuring inference latency\n")

batch_size = 1
measure_inference_latency(1)
batch_size = 12
measure_inference_latency(12)
print()
del model

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# function to measure training latency
def measure_training_throughput(batch_size, max_iters=5):
    times = []
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    # Do an extra iteration: first iteration is slower as get_batch doesn't overlap with backward pass
    for _ in range(max_iters + 1):
        with ctx:
            logits, loss = model(X, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        # timing and logging
        t1 = time.time()
        times.append(t1 - t0)
        t0 = t1
    training_total_seconds = sum(times[1:])
    tokens_per_iter = ddp_world_size * batch_size * block_size
    training_tokens_per_second = tokens_per_iter / training_total_seconds
    print(f"Training throughput, batch size {batch_size}: {training_tokens_per_second:.4f} tokens/second")
    
# print("\nMeasuring training throughput\n")

# batch_size = 4
# measure_training_throughput(batch_size)
# batch_size = 12
# measure_training_throughput(batch_size)

if ddp:
    destroy_process_group()
