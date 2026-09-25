# 拼圖 v6 枚舉實驗（baseline / shape / learned_shape）

## 設計
- D4 對稱封鎖：先封鎖已知解，每找到一解再封鎖其 D4
- 各方法連續找 **5** 解
- **learned_shape (v123)**：L0 cascade + @25% oracle（`ranker_train_v1_v2_v3.joblib`）
- **learned_shape (v12345)**：同上設定，模型 `ranker_train_v1_v2_v3_v4_v5.joblib`（見 `../enum_v2_v12345/`）

> 枚舉各方法解序列不同，跨方法平均時間僅供參考；**主 KPI 見 D4 next-sol**（`baselines/v6/l0_benchmark/`）。

## CNF 規模（base）
| 方法 | Placements | Base 子句數 | build (s) |
|------|------------|-------------|-----------|
| baseline | 7,432 | 65,393,099 | 40.972 |
| shape | 7,432 | 67,500,467 | 1864.302 |
| learned_shape (v123) | 7,432 | 66,106,197 | 1268.821 |
| learned_shape (v12345) | 7,432 | 66,087,849 | 958.359 |

## 平均求解時間
| 方法 | 各次 (s) | 平均 (s) | 標準差 |
|------|----------|----------|--------|
| baseline | [79.578, 36.097, 52.297, 27.276, 40.542] | **47.158** | 20.241 |
| shape | [39.452, 120.671, 31.935, 30.483, 27.622] | **50.033** | 39.729 |
| learned_shape (v123) | [44.645, 54.961, 34.385, 47.531, 28.306] | **41.966** | 10.619 |
| learned_shape (v12345) | [41.234, 62.110, 26.816, 54.451, 21.329] | **41.188** | 17.425 |

- shape / baseline：**≈ 0.94×**
- learned v123 / baseline：**≈ 1.12×**
- learned v123 / shape：**≈ 1.19×**
- learned v12345 / learned v123：**≈ 1.02×**（幾乎相同）

## 與 D4 next-sol 對照（v6，10 seed）

| 指標 | learned v123 | learned v12345 | full shape |
|------|--------------|----------------|------------|
| D4 next-sol 平均 | 43.2 s | **31.2 s** | 32.3 s |
| 枚舉 5 解平均 | 41.97 s | 41.19 s | 50.03 s |

## 驗證
- baseline: 5/5 解通過 no-touch
- shape: 5/5 解通過 no-touch
- learned_shape (v123): 5/5 解通過 no-touch
- learned_shape (v12345): 5/5 解通過 no-touch

## 詳細說明
- 三方法 v123 枚舉：`results.json`（本目錄）
- v12345 learned-only：`../enum_v2_v12345/results.json`
- 實驗詳解：`../../learned_shape/EXPERIMENTS_DETAILED.md` §10.4、§11.3
