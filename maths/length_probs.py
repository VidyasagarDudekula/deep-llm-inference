import torch
from maths.sequence_score import sequence_logprob, compute_nll

gen1 = torch.manual_seed(42)
gen2 = torch.manual_seed(43)
logits_a = torch.randn((10, 2, 50), generator=gen1)
target_ids_a = torch.randint(0, 50, (10, 2), generator=gen1)
logits_b = torch.randn((10, 10, 50), generator=gen2)
target_ids_b = torch.randint(0, 50, (10, 10), generator=gen2)

seq_probs_a = sequence_logprob(logits_a, target_ids_a)
seq_probs_b = sequence_logprob(logits_b, target_ids_b)

print("Test Quality using the sequence log prob for shorter sequence(seq_probs_a) and longer sequence (seq_probs_b):-")
print(f"seq_probs_a:- {seq_probs_a}\n seq_probs_b:- {seq_probs_b}")
print(f"Not coparable also, just because a sequence is long, the metrics made it look like it is in a bad generation.")

print("\n\n")
seq_nll_a = compute_nll(logits_a, target_ids_a)
seq_nll_b = compute_nll(logits_b, target_ids_b)

print("Test Quality using the NLL for shorter sequence(seq_nll_a) and longer sequence (seq_nll_b):-")
print(f"seq_nll_a:- {seq_nll_a}\n seq_nll_b:- {seq_nll_b}")
print(f"this is reasonable, as it is giving some comparable metrics regardless of the length of the genration.")
print(f"Also this is min of log prob will give overall view of a generation for a sequence.")



