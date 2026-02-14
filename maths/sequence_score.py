import torch
import torch.nn as nn
import torch.nn.functional as F
from maths.softmax import stable_log_softmax
import math

gen = torch.manual_seed(42)
batch_size = 4
seq_len = 10
vocab_size = 50


def sequence_logprob(logits: torch.Tensor, target_ids: torch.Tensor):
    f_sft = stable_log_softmax()
    log_probs = f_sft(logits)
    out = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    out = torch.sum(out, dim=-1)
    return out

def compute_nll(logits: torch.Tensor, target_ids: torch.Tensor):
    B, T, C = logits.shape
    out = sequence_logprob(logits, target_ids)
    out = torch.mean(out/T, dim=-1)
    return -out


def test_seq_log_probs():
    logits = torch.randn((batch_size, seq_len, vocab_size), generator=gen)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    return sequence_logprob(logits, target_ids)

def print_sequence_report(logits: torch.Tensor, target_ids: torch.Tensor):
    stable_sft = stable_log_softmax()
    probs = stable_sft(logits)
    running_log_prob = 0.0
    for i in range(target_ids.shape[-1]):
        current_prob = probs[0, i, target_ids[0, i]].item()
        running_log_prob += current_prob
        print(f"Token id:- {target_ids[0, i].item()}, log_prob:- {current_prob}, cumm_log_prob:- {running_log_prob}, probability:- {math.exp(running_log_prob)*100}%")
    

def test_order():
    logits = torch.randn((1, 3, vocab_size), generator=gen)
    target_a = torch.tensor([10, 20, 30], dtype=torch.int64).view(1, 3)
    target_b = torch.tensor([30, 10, 20], dtype=torch.int64).view(1, 3)
    score_a = sequence_logprob(logits, target_a).item()
    score_b = sequence_logprob(logits, target_b).item()
    print(score_a, score_b)

def test_nll_loss():
    logits = torch.randn((batch_size, seq_len, vocab_size), generator=gen)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    f_sft = stable_log_softmax()
    probs = f_sft(logits)
    nll = F.nll_loss(probs.view(-1, vocab_size), target_ids.view(-1))
    nll_custome = compute_nll(logits, target_ids)
    print(nll, nll_custome)



if __name__ == '__main__':
    logits = torch.randn((1, seq_len, vocab_size), generator=gen)
    target_ids = torch.randint(0, vocab_size, (1, seq_len), generator=gen)
    print_sequence_report(logits, target_ids)
