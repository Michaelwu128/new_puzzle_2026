# 拼圖 v5 枚舉實驗（baseline / shape / learned_shape）

## 設計
- D4 對稱封鎖：先封鎖已知解，每找到一解再封鎖其 D4
- 各方法連續找 **5** 解
- **learned_shape**：L0 cascade + @25% oracle（v123 ranker）

## CNF 規模（base）
| 方法 | Placements | Base 子句數 | build (s) |
|------|------------|-------------|-----------|
| baseline | 7,232 | 60,727,023 | 27.629 |
| shape | 7,232 | 62,267,303 | 1810.627 |
| learned_shape | 7,232 | 61,253,998 | 1195.398 |

## 平均求解時間
| 方法 | 各次 (s) | 平均 (s) | 標準差 |
|------|----------|----------|--------|
| baseline | [12.307, 17.127, 11.913, 18.716, 11.104] | **14.233** | 3.441 |
| shape | [24.778, 24.174, 17.169, 15.521, 26.394] | **21.607** | 4.906 |
| learned_shape | [3.532, 3.819, 4.492, 5.793, 3.69] | **4.265** | 0.929 |

- shape / baseline：**≈ 0.66×**
- learned / baseline：**≈ 3.34×**
- learned / shape：**≈ 5.07×**

## 驗證
- baseline: 5/5 解通過 no-touch
- shape: 5/5 解通過 no-touch
- learned_shape: 5/5 解通過 no-touch