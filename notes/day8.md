## Notes — Why we scale attention by `sqrt(dk)` (variance proof + what it means for softmax)

### 1) Why we normalize embeddings (mean=0, var=1)
- For Q and K, each token embedding (per head) is on a common scale: mean ~0 and variance ~1.
- This doesn’t “destroy” the info — it mainly changes the **units / scale** so one token (or one batch) doesn’t dominate just because its numbers are larger.
- If one embedding has variance ~100 and another has variance ~1.2, even with similar gradients, their impact is very different. Normalizing makes training + gradients behave more predictably.

---

### 2) Setup for the dot-product (single attention score)
Let a single attention score be:

- `S = sum_{i=1..dk} (q_i * k_i)`

Assumptions:
- `q_i` and `k_i` are independent
- `E[q_i] = E[k_i] = 0`
- `Var(q_i) = Var(k_i) = 1`

---

### 3) Micro proof: why `Var(q_i * k_i) = 1`
Use:
- `Var(X) = E[X^2] - (E[X])^2`
- Here `X = q_i k_i`

Step-by-step:
- `E[q_i k_i] = E[q_i] * E[k_i] = 0 * 0 = 0`  (independent)
- `E[(q_i k_i)^2] = E[q_i^2 k_i^2] = E[q_i^2] * E[k_i^2]` (independent)
- Since `Var(q_i) = 1` and `E[q_i] = 0`, we have `E[q_i^2] = Var(q_i) + (E[q_i])^2 = 1 + 0 = 1`
  - same for `k_i`: `E[k_i^2] = 1`
- So `E[(q_i k_i)^2] = 1 * 1 = 1`
- Therefore:
  - `Var(q_i k_i) = 1 - 0 = 1`

This is the exact missing proof for the assumption I used earlier.

---

### 4) Why the dot-product variance becomes `dk`
Now for:
- `S = (q_1 k_1) + (q_2 k_2) + ... + (q_dk k_dk)`

Because we are summing **dk independent terms**, and each term has variance `1`:
- `E[S] = sum E[q_i k_i] = 0`
- `Var(S) = sum Var(q_i k_i) = 1 + 1 + ... (dk times) = dk`

This is the core “why”: **each extra dimension adds another independent source of randomness**, so the total spread grows linearly with `dk`.

---

### 5) Why divide by `sqrt(dk)`
- Std is `sqrt(Var)`, so `std(S) = sqrt(dk)`
- If we define `S_scaled = S / sqrt(dk)`, then:
  - `Var(S_scaled) = Var(S) / (sqrt(dk))^2 = dk / dk = 1`

So scaling brings the score distribution back to a stable unit variance before softmax.

---

### 6) What this changes in softmax + training stability
- Without scaling, attention logits can get very large (variance = dk), so softmax becomes **too spiky** (almost one-hot).
- Spiky softmax means the model “hard locks” onto a token, and gradients become unstable / poorly behaved.
- With scaling, logits stay in a reasonable range → softmax stays “healthy” → gradients don’t collapse.

---

## Graph Notes

### A) Sequence length vs time compute (Attention)
- The plot shows attention time growing fast as sequence length increases.
- This matches the core cost: attention builds a `(T, T)` score matrix, so compute/memory pressure rises heavily as T grows.

### B) Entropy and variance vs dimensions
- The variance curve increases roughly linearly with dimension (this is the `Var(S) = dk` story showing up visually).
- The entropy curve looks almost flat in this plot mainly because it’s on a very different numerical scale than variance (so it gets visually crushed). If I want to compare them properly, I should plot entropy on a second axis or separate figure.