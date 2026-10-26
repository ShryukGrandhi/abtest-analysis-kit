### Fixed-horizon test, 40000 users, 1000 simulated experiments

| metric | A/A false positive rate (raw) | A/A FPR (CUPED) | power at +10% conversion (raw) | power (CUPED) |
|---|---|---|---|---|
| orders | 6.4% (5.0%-8.1%) | 5.6% (4.3%-7.2%) | 83.0% (80.5%-85.2%) | 95.9% (94.5%-97.0%) |
| revenue | 5.1% (3.9%-6.6%) | 5.6% (4.3%-7.2%) | 65.9% (62.9%-68.8%) | 80.7% (78.1%-83.0%) |

Mean CUPED variance reduction (pre-period covariate): sessions 36.0%, orders 36.0%, revenue 27.1%

### Peeking: 20 interim looks, 500 users/arm per look, 2000 simulations

| truth | naive z-test, stop at first p<0.05 | mSPRT always-valid p | single test at the end | mSPRT mean stopping look |
|---|---|---|---|---|
| null | 26.1% (24.2%-28.1%) | 1.1% (0.7%-1.6%) | 5.0% (4.1%-6.0%) | - |
| effect_0.05sd | 97.0% (96.2%-97.7%) | 73.9% (71.9%-75.8%) | 94.8% (93.7%-95.7%) | 11.3 |

### 95% CI coverage for relative revenue lift (A/A, 2000 users, 500 simulations)

| method | coverage | mean CI width |
|---|---|---|
| delta method | 94.6% (92.3%-96.3%) | 71.0% |
| Poisson bootstrap (1000 replicates) | 95.6% (93.4%-97.1%) | 72.7% |

### Sample ratio mismatch check (200000 users, p < 0.001, 5000 simulations)

- flagged when split is truly 50/50: 0.1% (0.1%-0.3%)
- flagged when split is 50.5/49.5: 87.8% (86.8%-88.7%)

### Multiple metrics: 20 metrics, 4 with a real effect, 1000 simulations

| correction | false discovery rate | P(any false positive) | power |
|---|---|---|---|
| none | 17.9% | 55.5% (52.4%-58.6%) | 76.8% |
| Benjamini-Hochberg | 4.0% | 11.9% (10.0%-14.1%) | 48.1% |
| Holm | 2.5% | 5.2% (4.0%-6.8%) | 37.5% |

### Pipeline throughput (200000 users, 1596943 event rows, 54.3 MB CSV, best of 3)

- CSV parse + validation + per-user aggregation: 1.26 s (1,269,320 rows/s)
- analysis of 5 metrics (CUPED, ratios, BH): 19.5 ms

Total benchmark time: 45.0 s (python 3.12.14, numpy 2.5.4, Darwin arm64)
