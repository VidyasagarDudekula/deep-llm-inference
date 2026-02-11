## Day Log — Stable Softmax & Stable Log-Softmax

Yesterday I implemented a numerically stable Softmax using the max-shift trick; today I added a stable Log-Softmax using `logsumexp` to keep results invariant under large input shifts.  
The Log-Softmax path is slightly slower (~0.054s vs ~0.045s on my test) because it performs extra reduction/log operations compared to plain Softmax.  
Trade-off is expected: a small compute cost for improved numerical safety and more reliable gradients at extreme magnitudes.
