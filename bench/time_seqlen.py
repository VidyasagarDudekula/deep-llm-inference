from toy_model.nn_layers import EmbeddingLayer, SelfAttention, ProjLayer
from config import device, get_async_time, device_synchronize
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt


def test_seq_len_size(vocab_size: int, dim: int, seq_len: int, num_head: int, head_dim: int, x: torch.Tensor):
    cal_times = 0.0
    iterations = 10
    for _ in range(iterations):
        start = get_async_time()
        end_attns = get_async_time()
        embed_layer = EmbeddingLayer(vocab_size=vocab_size, dim=dim).to(device)
        inputs = embed_layer(x)
        attns_module = SelfAttention(dim=dim, seq_len=seq_len, num_head=num_head, head_dim=head_dim).to(device)
        start.record()
        out = attns_module(inputs)
        end_attns.record()
        device_synchronize()
        attn_time = start.elapsed_time(end_attns)
        cal_times += attn_time
    return cal_times/iterations



if __name__ == '__main__':
    B = 10
    T = 256
    C = 512
    num_head = 4
    head_dim = C//num_head
    gen = torch.Generator(device)
    gen.manual_seed(42)
    seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    attn_times = []
    proj_times = []
    #warmup:-
    x = torch.randint(0, 500, (B, T), device=device, generator=gen)
    at = test_seq_len_size(500, C, T, num_head, head_dim, x)
    for sl in seq_lens:
        x = torch.randint(0, 100, (B, sl), device=device, generator=gen)
        at = test_seq_len_size(100, C, sl, num_head, head_dim, x)
        print(f"Seq_len:- {sl}, total time:- {at}")
        attn_times.append(at)
    
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, attn_times, label='Attns Time', color='blue')
    plt.xlabel('Seq_Len')
    plt.ylabel('Time')
    plt.title('Seqence length vs Time Compute')
    plt.legend()
    plt.grid(True)
    plt.savefig('Sequence_length_time_plot.png')
    print("Plot saved as Sequence_length_time_plot.png")
