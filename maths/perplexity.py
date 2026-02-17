import torch
from maths.sequence_score import compute_nll

def compute_perplexity(logits: torch.Tensor, target_ids: torch.Tensor):
    out = compute_nll(logits, target_ids)
    return torch.exp(out)

if __name__ == '__main__':
    gen = torch.manual_seed(42)
    cheating = torch.randn((2, 1, 50), generator=gen)
    cheating[0, 0, 10] = 25
    cheating[1, 0, 21] = 100
    target_ids = torch.tensor([[10], [21]])
    clueless = torch.ones((2, 1, 50))
    print(target_ids.shape, cheating.shape, clueless.shape)
    out_cheating = compute_perplexity(cheating, target_ids)
    out_clueless = compute_perplexity(clueless, target_ids)
    print(out_cheating, out_clueless)