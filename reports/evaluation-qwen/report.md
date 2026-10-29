# Text-to-SQL model evaluation

Dataset: `data\processed\gretel_finance_test.json` (72 examples)

| Model | Exact match | Valid SQL | Execution accuracy* | Loss | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.125 | 0.625 | 0.5208333333333334 | 0.4761077922941305 | 9208.079699998052 | 22468.838164999215 | 28249.393183000844 |
| fine_tuned | 0.222 | 0.681 | 0.5625 | 0.2348708281430586 | 17439.876000000368 | 38004.07390500004 | 42007.01046399946 |

* Execution accuracy uses executable reference examples only. Exact match ignores whitespace/comments and folds keyword case; it is not semantic SQL equivalence. Test loss scores target SQL tokens only; overlength examples are skipped and reported.