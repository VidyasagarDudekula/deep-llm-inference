import torch

device = torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

class ModelCfg:
    def __init__(self):
        self.max_seq_len = 128
        self.dim = 512
        self.num_head = 4
        assert self.dim % self.num_head == 0
        self.head_dim = self.dim//self.num_head

def get_async_time():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True)
    elif torch.backends.mps.is_available():
        return torch.mps.Event(enable_timing=True)
    return None

def device_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    return None