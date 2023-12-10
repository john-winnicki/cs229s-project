import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group


# ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
# if ddp:
init_process_group(backend='nccl')

# Example tensor on GPU 0
if dist.get_rank() == 0:
    tensor = torch.randn(2).to('cuda:0')
    print(f'Tensor {tensor} set on GPU {dist.get_rank()}')
else:
    tensor = torch.zeros(2).to('cuda:1')
    print(f'Tensor {tensor} initialized on GPU {dist.get_rank()}, check GPU {tensor.device}')

# Broadcasting tensor from GPU 0 to GPU 1
dist.broadcast(tensor, src=0)
print(f'Tensor {tensor} set from GPU {dist.get_rank()} on GPU {tensor.device}')

destroy_process_group()