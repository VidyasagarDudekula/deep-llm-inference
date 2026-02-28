from maths.softmax import stable_softmax
from maths.entropy import compute_entropy
from config import device
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt


class TestSelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
    
    def forward(self, x, scale=True):
        B, T, C = x.shape
        q = torch.randn((B, T, C), device=device)
        k = torch.randn((B, T, C), device=device)

        attn_logits = q @ k.transpose(-2, -1)
        if scale:
            attn_logits *= (C ** -0.5)

        return attn_logits



if __name__ == '__main__':
    dims = [64, 128, 256, 512, 1026, 2048]
    entropy_list = []
    var_list = []
    gen = torch.Generator(device)
    gen.manual_seed(42)
    stb_sft = stable_softmax()
    for dk in dims:
        attn_module = TestSelfAttention(dim=dk).to(device)
        x = torch.randn((1, 10, dk), device=device, generator=gen)
        attn_logits = attn_module(x, False)
        probs = stb_sft(attn_logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)
        entropy_value = compute_entropy(attn_logits).item()
        entropy_list.append(entropy_value)
        var_value = torch.var(attn_logits, unbiased=False).item()
        var_list.append(var_value)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, entropy_list, label='Entropy', color='blue')
    plt.plot(dims, var_list, label='Variance', color='orange')
    plt.xlabel('Dimentions')
    plt.ylabel('Change')
    plt.title('Entropy and Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig('Variance_dk_plot.png')
    print("Plot saved as Variance_dk_plot.png")
