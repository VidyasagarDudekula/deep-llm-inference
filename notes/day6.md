## Day 06 — Perplexity from scratch (and why it’s just exp of avg NLL)

- Perplexity is basically **`exp(avg NLL)`**:
  - `PPL = exp( - (1/T) Σ logp_t )`
  - So it is telling **how surprised the model is on the true tokens**, averaged over the sequence.
  - Intuition: **higher PPL = more confusion about the truth**.

- I connected this with **entropy**:
  - `exp(entropy)` is like the model’s **effective guessing count** *without needing target ids*.
  - So **entropy = model’s internal uncertainty**, while **PPL = surprise on the real label**.

- Important difference (the “confident hallucination” case):
  - Model can be **very confident but wrong** → **low entropy**, but **high NLL / high PPL**.
  - So entropy tells *“am I confused?”* and PPL tells *“am I right about the truth?”*.

- Random test helped me see why PPL can go above vocab size:
  - With random logits, the model creates random spikes and focuses on fewer tokens (example: `exp(entropy) ~ 31.6` for vocab 50).
  - But target ids are random too, so the true token often falls in the “ignored” tokens → tiny probability → huge penalty.
  - That’s why **PPL can become worse than uniform guessing** (example: `exp(NLL) ~ 85.9` even though vocab is 50).

- Verified the two extremes (this is the clean mental ruler):
  - **Clueless model (uniform over V):** `NLL = log(V)` → `PPL = V` (here: **50**).
  - **Perfect / cheating model (prob=1 on true token):** `NLL → 0` → `PPL → 1`.

- Note to self:
  - PPL is strong for “how well the model predicts the dataset truth”, but it can be misleading for decoding quality when comparing **short vs long** generations (length effects, averaging effects, etc.).
