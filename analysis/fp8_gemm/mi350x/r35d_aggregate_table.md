## CRR vs RRR — per-cell paired bench (BABA, n=10/kernel)

| Cell | Shape | GPU6 Δ% | GPU6 t | GPU6 adv | GPU7 Δ% | GPU7 t | GPU7 adv | min Δ% | Adv expected | Verdict |
|---|---|---:|---:|:---:|---:|---:|:---:|---:|:---:|---|
| 8B-Q | 4096x4096x4096 | +4.83 | +2.71 | 0 | n/a | n/a | n/a | +4.83 | no | OK (no adv) |
| 8B-K | 4096x1024x4096 | +7.74 | +24.14 | 1 | n/a | n/a | n/a | +7.74 | YES | **SHIP** |
| 8B-V | 4096x1024x4096 | +8.57 | +15.43 | 1 | n/a | n/a | n/a | +8.57 | YES | **SHIP** |
| 8B-O | 4096x4096x4096 | +4.73 | +2.34 | 0 | n/a | n/a | n/a | +4.73 | no | OK (no adv) |
| 8B-Gate | 4096x14336x4096 | +6.25 | +6.82 | 0 | n/a | n/a | n/a | +6.25 | no | OK (no adv) |
| 8B-Up | 4096x14336x4096 | +5.26 | +10.49 | 0 | n/a | n/a | n/a | +5.26 | no | OK (no adv) |
| 8B-Down | 4096x4096x14336 | +9.51 | +2.77 | 0 | n/a | n/a | n/a | +9.51 | no | OK (no adv) |
| 70B-Q | 4096x8192x8192 | +7.72 | +6.80 | 0 | n/a | n/a | n/a | +7.72 | no | OK (no adv) |
| 70B-K | 4096x1024x8192 | +11.06 | +28.08 | 1 | n/a | n/a | n/a | +11.06 | YES | **SHIP** |
| 70B-V | 4096x1024x8192 | +10.21 | +4.01 | 1 | n/a | n/a | n/a | +10.21 | YES | **SHIP** |
| 70B-O | 4096x8192x8192 | +6.94 | +9.61 | 0 | n/a | n/a | n/a | +6.94 | no | OK (no adv) |
| 70B-Gate | 4096x28672x8192 | +7.93 | +41.49 | 1 | n/a | n/a | n/a | +7.93 | YES | **SHIP** |
| 70B-Up | 4096x28672x8192 | +7.87 | +39.57 | 1 | n/a | n/a | n/a | +7.87 | YES | **SHIP** |
| 70B-Down | 4096x8192x28672 | +12.16 | +30.31 | 1 | n/a | n/a | n/a | +12.16 | YES | **SHIP** |

## CRR vs RCR — square Q/O cells (3-way comparison)

| Cell | Shape | GPU6 Δ% (RCR vs CRR) | GPU6 t | GPU7 Δ% | GPU7 t | min Δ% (RCR>CRR) |
|---|---|---:|---:|---:|---:|---:|
| 8B-Q | 4096x4096x4096 | +7.05 | +4.22 | n/a | n/a | +7.05 |
| 8B-O | 4096x4096x4096 | +5.83 | +3.50 | n/a | n/a | +5.83 |
| 70B-Q | 4096x8192x8192 | +8.32 | +8.26 | n/a | n/a | +8.32 |
| 70B-O | 4096x8192x8192 | +8.20 | +9.16 | n/a | n/a | +8.20 |

## sclk-post-preheat (R34 contention diagnosis)

| Cell | GPU6 sclk MHz | GPU7 sclk MHz |
|---|---:|---:|
| 8b_q | 2036 | n/a |
| 8b_k | 2294 | n/a |
| 8b_v | 2317 | n/a |
| 8b_o | 2299 | n/a |
| 8b_gate | 2306 | n/a |
| 8b_up | 2309 | n/a |
| 8b_down | 1895 | n/a |
| 70b_q | 2243 | n/a |
| 70b_k | 2314 | n/a |
| 70b_v | 2340 | n/a |
| 70b_o | 2285 | n/a |
| 70b_gate | 2256 | n/a |
| 70b_up | 2222 | n/a |
| 70b_down | 2273 | n/a |

## Correctness summary

All correctness checks PASS (snr ≥ 48 dB, pass_rate=100%, det_ok=True).