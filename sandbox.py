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
    dist.broadcast(tensor, src=0)
    tensor2 = torch.randn(2).to('cuda:0')
    print(f'Tensor {tensor2} set on GPU {dist.get_rank()}')
    dist.broadcast(tensor2, src=0)
else:
    a = torch.zeros(2).to('cuda:1')
    dist.broadcast(a, src=0)
    print(f'Tensor {a} sent to GPU {dist.get_rank()}, check GPU {a.device}')
    b = torch.zeros(2).to('cuda:1')
    dist.broadcast(b, src=0)
    print(f'Tensor {b} sent to GPU {dist.get_rank()}, check GPU {b.device}')

destroy_process_group()