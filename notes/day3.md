## Day 3 Log — Sequence Score (Stable Log-Softmax)

### Run Output (Token-wise)
| Step | Token ID | Log Prob | Cumulative Log Prob | Sequence Prob (%) |
|---:|---:|---:|---:|---:|
| 1 | 37 | -4.9223976135 | -4.9223976135 | 0.7281651315% |
| 2 | 39 | -4.8755450249 | -9.7979426384 | 0.0055565801% |
| 3 | 29 | -3.4756174088 | -13.2735600471 | 0.0001719357% |
| 4 | 7  | -4.7103977203 | -17.9839577675 | 0.0000015476% |
| 5 | 35 | -5.5079007149 | -23.4918584824 | 0.0000000063% |
| 6 | 38 | -3.4334130287 | -26.9252715111 | 0.0000000002% |
| 7 | 17 | -4.2915420532 | -31.2168135643 | 0.000000000003% |
| 8 | 14 | -6.2232484818 | -37.4400620461 | 0.0000000000000055% |
| 9 | 7  | -4.2397117615 | -41.6797738075 | 0.000000000000000079% |
| 10| 24 | -4.1868600845 | -45.8666338921 | 0.0000000000000000012% |

### Notes
- Token **14** is the biggest spike (most negative log-prob), so it’s the point of **maximum surprise / confusion** given the prefix context.
- **Negative log-likelihood (NLL)** tells, on average, how probable the model thinks the correct tokens are for a given sequence.
- **Higher NLL ⇒ lower likelihood** for the correct prediction, i.e., the model is **more uncertain / more confused**.