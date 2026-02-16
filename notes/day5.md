## Day Notes — Entropy as “Uncertainty” + Temperature Effect

- I computed **entropy** from the model distribution using `H(p) = -sum(p * log p)`. This is a direct measure of how **uncertain / spread out** the predictions are.
- For a **uniform** logits case (all ones over 50 tokens), entropy came out **~3.912**, and `exp(entropy) ≈ 50`, which matches the idea of **“effective guessing count”** (the model is basically equally guessing among 50 tokens).
- For the **peaked** logits case (two tokens boosted to 100), entropy came out **~0.6931**, and `exp(entropy) ≈ 2`, meaning the model is effectively choosing between **2 tokens** (high confidence / low uncertainty).
- Temperature test on the peaked logits:
  - For small to normal temps (0.01 → ~5), entropy stayed around **0.6931** (still very peaked on those 2 tokens).
  - When temperature got large (10 → 100), entropy increased (**0.7063 → 2.3129 → 3.7734 → 3.8777**) because higher temperature makes the distribution **more uniform**, so the model becomes **less confident** and spreads probability to more tokens.
