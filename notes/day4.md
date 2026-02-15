## Day Log — Length Bias, NLL vs Log-Prob, and Chatbot Filtering Ideas

### Run
**Command:** `python -m maths.length_probs`

### Test 1 — Quality via raw sequence log-prob (length-biased)
**Goal:** compare a shorter sequence (`seq_probs_a`) vs a longer sequence (`seq_probs_b`)

- `seq_probs_a` (short):
  - tensor([ -9.5571,  -6.9069, -12.4230,  -8.7082, -11.6055, -10.6919,  -9.3203, -10.5201,  -8.8899,  -6.3690])

- `seq_probs_b` (long):
  - tensor([-46.3163, -44.3665, -44.4756, -40.5072, -49.0028, -42.1155, -37.8311, -42.5875, -46.1512, -46.3185])

**Note:** Not comparable. Just because a sequence is long, this metric makes it look like it is in a bad generation.

---

### Test 2 — Quality via NLL (length-normalized / comparable)
**Goal:** compare shorter (`seq_nll_a`) vs longer (`seq_nll_b`) using NLL

- `seq_nll_a`: **4.749589443206787**  
- `seq_nll_b`: **4.396721363067627**

**Note:** This is reasonable, because it gives comparable metrics regardless of the length of the generation.  
Also, this (kind of) gives an overall view of a generation for a sequence.

---

## Additional Analysis — Chatbot Filtering / Content Control

If I’m building a chatbot and I’m trying to filter out the contents, at this point I have **2 metrics** to consider to test generation quality (assuming these are the ones we use to measure):

1. **NLL (Negative Log Likelihood)**  
   - Gives how quality the model generation for a given response is, but it is on average.  
   - Idea: set a threshold (example: max **2**). If NLL for a sequence goes above that, I might **filter/flag** it for safety and try generating again with different decoding/sampling.

2. **Min Log-Prob (token-level control)**  
   - Token level control idea: check the **min log-prob** for every **5 tokens**.  
   - If it goes less than a threshold, I might **discard the last 5 tokens** and generate another set of 5 continuous tokens.  
   - This seems better than discarding the entire sequence, but I’m not sure I’m capturing the overall quality of the sequence still.

### Follow-up
Point 2 (backtracking and regenerating just the bad segment) is a sophisticated technique used in some constrained generation systems — **but there is a catch**:

If I discard the last 5 tokens because they were "bad," and then I ask the model to generate again from the **same context**, what is likely to happen?

### Answer / My Plan
- Yes, the decoding sampling will be changed — I don’t use exact settings, but the **context is still the same**.  
- Or maybe I will call a **safety model / draft model** to generate for 5 tokens (to guide in a safety regulations following order), and later continue with the original model again.

