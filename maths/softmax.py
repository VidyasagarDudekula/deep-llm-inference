import torch
import torch.nn as nn
from config import device



class naive_softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, dim=-1):
        exp_values = torch.exp(x)
        sum_values = torch.sum(exp_values, dim=dim, keepdim=True)
        out = exp_values/sum_values
        return out


class stable_softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, dim=-1):
        max_values = torch.max(x, dim=dim, keepdim=True).values
        x_shifted = x - max_values
        exp_values = torch.exp(x_shifted)
        sum_values = torch.sum(exp_values, dim=dim, keepdim=True)
        out = exp_values/sum_values
        return out


class stable_log_softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, dim: int=-1):
        max_values = torch.max(x, dim=dim, keepdim=True).values
        shifted_x = x - max_values
        log_probs = shifted_x - torch.logsumexp(shifted_x, dim=dim, keepdim=True)
        return log_probs
    
def get_next_token(x: torch.Tensor):
    m = stable_softmax()
    x = m(x, -1)
    indices = torch.argmax(x, dim=-1, keepdim=True)
    return indices
    


if __name__ == '__main__':
    m = stable_softmax()
    gen = torch.manual_seed(42)
    i = 0
    logits = torch.randn((1, 50), requires_grad=True, generator=gen)
    probs = m(logits, -1)
    sorted_tensor = torch.sort(probs, dim=-1, descending=True)
    sorted_probs = sorted_tensor.values
    sorted_indices = sorted_tensor.indices
    cumm_sum = torch.cumsum(sorted_probs, dim=-1)
    top_k = sorted_probs[:, :10]
    top_k_sum = cumm_sum[:, :10]
    top_k_indices = sorted_indices[:, :10]
    top_logits = torch.gather(logits, dim=-1, index=sorted_indices)[:, :10]
    print("|  Rank  |  Logits  |  probs  |  token_id  |  cumsum  |")
    for i in range(10):
        print(f"|  {i+1}  |  {top_logits[0, i]:.2f}  |  {top_k[0, i]:.2f}  |  {top_k_indices[0, i]:.2f}  |  {top_k_sum[0, i]:.2f}  |")
    
    