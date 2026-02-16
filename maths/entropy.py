import torch
from maths.softmax import stable_log_softmax, stable_softmax

stb_sft = stable_softmax()
stb_log_sft = stable_log_softmax()

def compute_entropy(logits: torch.Tensor):
    probs = stb_sft(logits, dim=-1)
    log_probs = stb_log_sft(logits, dim=-1)
    out = - torch.sum(probs * log_probs, dim=-1)
    return out


if __name__ == '__main__':
    gen = torch.manual_seed(42)
    logits_a = torch.ones((1, 50))
    logits_b = torch.ones((1, 50))
    logits_b[0, 4], logits_b[0, 23] = 100, 100
    out_a = compute_entropy(logits_a)
    out_b = compute_entropy(logits_b)
    print("Entropy:- ", out_a, out_b)
    print("take a look at the guessing count:- ")
    print("Exp of entropy:- ", torch.exp(out_a), torch.exp(out_b))
    
    # Test the temp vs emtropy:-
    temp_values = [0.01, 0.1, 0.2, 0.5, 0.6, 0.8, 1.0, 5.0, 10.0, 25.0, 60.0, 100.0]
    for t in temp_values:
        print(compute_entropy(logits_b/t))