# v6 CNF build 公平對照（同一時段連續跑）

**run_at** = 2026-06-18T09:46:35.858270+08:00
**puzzle** = v6
**learned_model** = `baselines/learned_shape/ranker_train_v1_v2_v3_v4_v5.joblib`

| 方法 | build_cnf (s) | write_cnf (s) | 合計 (s) | clauses |
|------|---------------|---------------|----------|---------|
| baseline | 31.681 | 12.65 | 44.331 | 65,393,099 |
| shape | 1314.733 | 20.368 | 1335.101 | 67,500,467 |
| learned_shape | 972.211 | 19.077 | 991.288 | 66,087,849 |

## 歷史對照

break-even: baseline 31.5s, learned 938.3s, shape 1232.5s (build only); l0 manifest: baseline 42.1s, learned 950.6s, shape 2061.4s.