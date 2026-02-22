## Day 07 — Self-Attention → Full toy LM pipeline + vocab bottleneck benchmark

- I clarified the attention shapes:
  - Q, K, V are (T, D) (or in batch: (B, T, D)), and `QK^T` becomes (T, T).
  - Each cell (i, j) is basically the similarity score between **token i as query** and **token j as key**.
  - Each row i is “what token i is looking at” across all tokens.

- I learned why we scale by `sqrt(dk)` before softmax:
  - If Q and K entries are like standard normal, the dot product variance grows with **D** (I verified with a small simulation: variance ≈ D).
  - Without scaling, attention scores become too spread out → softmax becomes too spiky → gradients get bad.
  - Dividing by `sqrt(dk)` brings variance back near 1, so softmax stays in a healthy range.

- I implemented multi-head self-attention and masking:
  - Multi-head is mostly reshaping and transposing, not a new concept.
  - Causal mask: set future positions to `-inf` so softmax makes them exactly 0 (no looking ahead).
  - Also supported padding mask for pad tokens.

- I understood what `A @ V` is doing:
  - After softmax, `A` rows sum to 1 (a probability distribution).
  - Output token vector = weighted sum of all value vectors (a “mix” of information from other tokens).

- I built the “sandwich” to make it a tiny language model forward pass:
  - **Embedding layer** converts token IDs (B, T) → vectors (B, T, D).
  - **Projection layer** converts (B, T, D) → logits (B, T, V).
  - Important fix: don’t put softmax inside the projection layer, because training loss (cross entropy) and decoding (temperature, sampling) want raw logits.

- Benchmark lesson (systems mindset):
  - My first timing was wrong because GPU ops are async (CPU was only measuring dispatch time).
  - After adding synchronization, the pattern became clear:
    - Attention time is basically **independent of vocab size**.
    - Projection time increases with vocab size, and it looks **roughly linear** (bigger V → more matmul work).
  - This explains why large vocab sizes (like 128k+) can make the final projection a real bottleneck.