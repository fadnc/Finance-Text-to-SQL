# Text-to-SQL model evaluation

Dataset: `data\processed\gretel_finance_test.json` (72 examples)

| Model | Exact match | Valid SQL | Execution accuracy* | Loss | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.000 | 0.458 | 0.2708333333333333 | 0.838281695329292 | 7449.542900001688 | 10360.683474997495 | 10686.572686990985 |
| fine_tuned | 0.167 | 0.653 | 0.5833333333333334 | 0.7738588099073491 | 1851.1532000047737 | 4046.1388549993 | 5359.200078008149 |

* Execution accuracy uses executable reference examples only. Exact match ignores whitespace/comments and folds keyword case; it is not semantic SQL equivalence. Test loss scores target SQL tokens only; overlength examples are skipped and reported.