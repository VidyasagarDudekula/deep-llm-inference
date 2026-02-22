import torch
import torch.nn as nn
import torch.nn.functional as F
from maths.softmax import stable_softmax
from config import device, ModelCfg


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embddings = nn.Embedding(self.vocab_size, self.dim)
    
    def forward(self, x):
        return self.embddings(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, seq_len: int, num_head:int, head_dim: int, is_causal: bool=True):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.q_w = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.k_w = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.v_w = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.w_o = nn.Linear(self.num_head * self.head_dim, self.dim)
        self.stb_sft = stable_softmax()
        self.register_buffer('tril', torch.tril(torch.ones((seq_len, seq_len)).bool()).unsqueeze(0).unsqueeze(1))
    
    def forward(self, x, mask = None):
        B, T, C = x.shape
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.v_w(x)

        q = q.contiguous().view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(B, T, self.num_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.is_causal:
            attn = attn.masked_fill(~self.tril[:, :, :T, :T], value=float('-inf'))
        
        if mask is not None:
            if mask.dim()==2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(~mask, value=float('-inf'))
        
        attn_scores = self.stb_sft(attn, dim=-1)

        out = attn_scores @ v
        out = out.transpose(1, 2).reshape(B, T, self.num_head*self.head_dim)
        out = self.w_o(out)

        return out


class ProjLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.proj = nn.Linear(self.dim, self.vocab_size)
    
    def forward(self, x):
        return self.proj(x)



if __name__ == '__main__':
    B = 10
    T = 256
    C = 512
    VOCAB_SIZE = 1000
    num_head = 4
    head_dim = C//num_head
    embedding_layer = EmbeddingLayer(VOCAB_SIZE, C).to(device)
    gen = torch.Generator(device)
    gen.manual_seed(42)
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device, generator=gen)
    print(f"Initial input shape:- {x.shape}")

    x = embedding_layer(x)
    print(f"input shape after embedding laer:- ", x.shape)

    attns_module = SelfAttention(dim=C, seq_len=T, num_head=num_head, head_dim=head_dim).to(device)
    out = attns_module(x)
    print(f"Data shape after the attention layer:- {out.shape}")

    proj_layer = ProjLayer(dim=C, vocab_size=VOCAB_SIZE).to(device)
    final_output = proj_layer(out)
    print(f"Proj Layer output Shape:- {final_output.shape}")