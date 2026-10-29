# Model comparison - finance held-out test set

Dataset: `data/processed/gretel_finance_test.json` (72 examples). Execution accuracy uses executable gold queries only.

| Model | Exact match | Valid SQL | Execution | Loss | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| tinyllama base | 1.4% | 79.2% | 43.8% | 0.724 | 21847 | 47418 |
| TinyLlama + Gretel finance | 18.1% | 73.6% | 56.2% | 0.482 | 9165 | 19884 |
| TinyLlama + multi-dataset | 18.1% | 65.3% | 54.2% | 0.487 | 3192 | 7945 |
| TinyLlama + multi-dataset (candidate) | 15.3% | 66.7% | 58.3% | 0.449 | 3377 | 8408 |
| TinyLlama + multi-dataset + Spider (ckpt359) | 16.7% | 63.9% | 54.2% | 0.460 | 3357 | 15555 |
| phi3-sql (TinyLlama adapter) | 4.2% | 31.9% | 22.9% | 0.935 | 2743 | 7097 |
| qwen base | 12.5% | 62.5% | 52.1% | 0.476 | 3032 | 8129 |
| Qwen2.5-Coder + WikiSQL/Spider/Finance | 22.2% | 68.1% | 56.2% | 0.235 | 5433 | 12587 |