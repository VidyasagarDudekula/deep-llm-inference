from toy_model.nn_layers import EmbeddingLayer, SelfAttention, ProjLayer
from config import device, get_async_time, device_synchronize
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt


def test_vocab_size(vocab_size: int, dim: int, seq_len: int, num_head: int, head_dim: int, x: torch.Tensor):
    start = get_async_time()
    end_attns = get_async_time()
    end_proj = get_async_time()
    satrt_proj = get_async_time()
    embed_layer = EmbeddingLayer(vocab_size=vocab_size, dim=dim).to(device)
    inputs = embed_layer(x)
    attns_module = SelfAttention(dim=dim, seq_len=seq_len, num_head=num_head, head_dim=head_dim).to(device)
    proj_layer = ProjLayer(dim=C, vocab_size=vocab_size).to(device)
    start.record()
    out = attns_module(inputs)
    end_attns.record()
    device_synchronize()
    satrt_proj.record()
    final_output = proj_layer(out)
    end_proj.record()
    device_synchronize()
    attn_time = start.elapsed_time(end_attns)
    proj_time = satrt_proj.elapsed_time(end_proj)
    return attn_time, proj_time



if __name__ == '__main__':
    B = 10
    T = 256
    C = 512
    num_head = 4
    head_dim = C//num_head
    gen = torch.Generator(device)
    gen.manual_seed(42)
    vocab_sizes = [500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 128000, 150000]
    attn_times = []
    proj_times = []
    #warmup:-
    x = torch.randint(0, 500, (B, T), device=device, generator=gen)
    at, pt = test_vocab_size(500, C, T, num_head, head_dim, x)
    for vs in vocab_sizes:
        x = torch.randint(0, vs, (B, T), device=device, generator=gen)
        at, pt = test_vocab_size(vs, C, T, num_head, head_dim, x)
        print(f"Vocab_size:- {vs}, total time:- {at + pt}")
        attn_times.append(at)
        proj_times.append(pt)
    
    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes, attn_times, label='Attns Time', color='blue')
    plt.plot(vocab_sizes, proj_times, label='Proj Time', color='orange')
    plt.xlabel('Vocab Size')
    plt.ylabel('Time')
    plt.title('Vocab Size vs Time Compute')
    plt.legend()
    plt.grid(True)
    plt.savefig('Vocab_size_time_plot.png')
    print("Plot saved as Vocab_size_time_plot.png")
