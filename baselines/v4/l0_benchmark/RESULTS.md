# v4 L0 benchmark 產物索引

**拼圖**：v4 hold-out | **模型（v123）**：`ranker_train_v1_v2_v3.joblib`

## CNF build 對照（`cnf_manifest.json`，2026-06-18 更新）

| tag | 模式 | top-K | L0 | build (s) | 額外 dead 子句 | sweep recall |
|-----|------|-------|-----|-----------|----------------|--------------|
| `baseline` | baseline | — | — | 37.1 | — | — |
| `shape` | shape 全掃 | — | — | 2436.8 | +1,630,496 | 100% |
| `learned_l0_k25` | v2-only ranker | 25% | 0.4 | 1808.5 | +604,265 | — |
| `learned_l0_k25_v123` | v123 合訓 | 25% | 0.4 | **999.2** | +566,171 | 34.7% |
| `learned_l0_k50_v123` | v123 合訓 | 50% | 0.4 | **1233.3** | +869,762 | 53.3% |
| `learned_k25_v123_no_l0` | v123 合訓 | 25% | — | **1195.5** | +687,057 | 42.1% |
| `learned_k50_v123_no_l0` | v123 合訓 | 50% | — | **1761.3** | +1,152,910 | 70.7% |

**vs shape（build）**：L0@25% **2.44×** 快；L0@50% **1.98×** 快；無 L0 @25% **2.04×**；無 L0 @50% **1.38×**。

## 端到端 JSON

| 檔案 | 內容 |
|------|------|
| `v4_l0_benchmark.json` | v2-only L0@25%（baseline + shape + learned） |
| `v4_l0_benchmark_v123.json` | v123 L0@25% |
| `v4_v123_no_l0_k25.json` | v123 無 L0 @25%（CNF + SAT，mean **28.31 s**） |
| `v4_v123_no_l0_k50.json` | v123 無 L0 @50%（CNF only） |
| `v4_v123_l0_k50.json` | v123 L0@50%（CNF only，build **1233.3 s**） |
| `v4_l0_compare_v2_vs_v123.md` | v2 vs v123 並排 + 無 L0 / L0@50% 對照 |

## 重現

```bash
cd encoding
bash run_v4_v123_no_l0_chain.sh   # 無 L0 @25% CNF+SAT → @50% CNF
python3 run_v4_l0_benchmark.py --puzzle v4 --cnf-only --learned-only \
  --top-k 50 --cascade-l0 0.4 \
  --model ../baselines/learned_shape/ranker_train_v1_v2_v3.joblib \
  --learned-tag learned_l0_k50_v123 \
  --report-name v4_v123_l0_k50.json --force-regen-cnf
python3 run_v4_l0_benchmark.py --all --learned-only \
  --model ../baselines/learned_shape/ranker_train_v1_v2_v3.joblib \
  --learned-tag learned_l0_k25_v123 \
  --report-name v4_l0_benchmark_v123.json
```

詳見 [`baselines/learned_shape/REPORT.md`](../../learned_shape/REPORT.md) §8.3–8.5。
