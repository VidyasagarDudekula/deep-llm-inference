import torch
import torch.nn as nn
from maths.softmax import stable_softmax, stable_log_softmax

stb_sft = stable_softmax()
stb_log_sft = stable_log_softmax()

def sample(logits: torch.Tensor, temperature: float=1.0, top_k: int=10, top_p: float=0.9):
    # top-k
    logits, logits_indices = torch.topk(logits, k=top_k, dim=-1)
    
    probs = stb_sft(logits/temperature, dim=-1)
    
    # top-p
    cumsum = torch.cumsum(probs, dim=-1)
    mask = cumsum <= top_p
    mask[:, 0] = True
    mask[:, 1:] |= (cumsum[:, :-1] <= top_p)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    indices = torch.multinomial(probs, num_samples=1)
    return torch.gather(logits_indices, dim=-1, index=indices)

if __name__ == '__main__':
    gen = torch.manual_seed(42)
    logits = torch.randn((2, 50), generator=gen)
    print(sample(logits, top_p=0.2, top_k=5))
     