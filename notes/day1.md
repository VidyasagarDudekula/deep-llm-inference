# Softmax + Temperature Notes (2026-02-09)

## What I did today
- Implemented **naive softmax** and **stable softmax** (subtract max for numerical stability).
- Ran a small experiment with random logits (`torch.manual_seed(42)`, vocab size = 50).
- Printed **top-10** tokens by probability + cumulative sum (to reason about top-k / top-p behavior).
- Tested **temperature scaling** by multiplying logits by 10 (equivalent to `softmax(logits / 0.1)`).

---

## Experiment 1: temperature T = 1.0 (original logits)

Command:
- `python -m maths.softmax`

Top-10 (approx):
| Rank | Logit | Prob | Token ID | Cumsum |
|---:|---:|---:|---:|---:|
| 1 | 1.93 | 0.08 | 0 | 0.08 |
| 2 | 1.84 | 0.07 | 45 | 0.15 |
| 3 | 1.68 | 0.06 | 23 | 0.21 |
| 4 | 1.65 | 0.06 | 9 | 0.26 |
| 5 | 1.64 | 0.06 | 16 | 0.32 |
| 6 | 1.49 | 0.05 | 1 | 0.37 |
| 7 | 1.38 | 0.04 | 47 | 0.41 |
| 8 | 1.33 | 0.04 | 27 | 0.45 |
| 9 | 1.30 | 0.04 | 25 | 0.50 |
| 10 | 1.28 | 0.04 | 24 | 0.53 |

### Notes / interpretation
- Top prob is only **~8%** → distribution is fairly **flat** (low confidence).
- The top few tokens have **similar probabilities**, meaning the model would be almost equally happy to choose among them.
- If the distribution stays this flat, **top-p (nucleus) sampling** with a high `p` (e.g., 0.9) could require **many tokens** to reach the threshold.

---

## Experiment 2: scale logits by 10 (temperature T = 0.1)

Math reminder:
- `softmax(c * logits)` is the same as `softmax(logits / T)` where `T = 1/c`.
- So multiplying by **10** ⇒ `T = 0.1` ⇒ *sharper / peakier distribution*.

Top-10 (approx):
| Rank | Logit | Prob | Token ID | Cumsum |
|---:|---:|---:|---:|---:|
| 1 | 19.27 | 0.60 | 0 | 0.60 |
| 2 | 18.45 | 0.26 | 45 | 0.86 |
| 3 | 16.81 | 0.05 | 23 | 0.91 |
| 4 | 16.49 | 0.04 | 9 | 0.95 |
| 5 | 16.42 | 0.03 | 16 | 0.99 |
| 6 | 14.87 | 0.01 | 1 | 0.99 |
| 7 | 13.84 | ~0.00 | 47 | 1.00 |
| 8 | 13.35 | ~0.00 | 27 | 1.00 |
| 9 | 12.96 | ~0.00 | 25 | 1.00 |
| 10 | 12.79 | ~0.00 | 24 | 1.00 |

### Notes / interpretation
- Huge change: top token jumps from **0.08 → 0.60**.
- Much more **confident / concentrated** distribution.
- For top-p, you now hit `p=0.9` with only about **3 tokens** (rank 3 gets to ~0.91).

---

## Key takeaways
- **Stable softmax** (subtract max) is the correct default: prevents overflow without changing results.
- **Temperature controls sharpness**:
  - Lower `T` (or multiplying logits) ⇒ more confident / peaky probabilities.
  - Higher `T` ⇒ flatter probabilities and more randomness.
- **Top-p cost/behavior**:
  - If probs are flat, nucleus set can become **large** (more tokens needed to reach `p`).
  - Sorting full vocab for top-p is expensive; might prefer **top-k first**, then top-p inside top-k, or other approximations.

---

## Next things to try
- Measure entropy vs temperature (`T ∈ {0.7, 1.0, 1.3}`).
- Implement actual **top-k sampling** and **top-p sampling** and log:
  - how many tokens are included for a fixed `p`
  - how output diversity changes across temperatures
- Compare `softmax` vs `log_softmax` usage for numerical stability in loss computations.
