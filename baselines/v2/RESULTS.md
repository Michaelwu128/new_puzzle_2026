# 拼圖 v2 枚舉實驗（baseline vs shape）

## 設計
- **baseline**：完整 placement 列舉 + no-touch，無針對性剪枝
- **shape**：與 baseline **相同放置集合**，僅加幾何死區（pairwise dead-pocket）子句
- **無** soft_prune、無 interior piece 豁免
- 每方法連續找 **5** 個解；每找到一解即封鎖其 **D4 旋轉／鏡射**（有效變換）
- 實驗開始前封鎖我畫的已知解及其對稱
- 報告 **kissat 平均時間**（含標準差）

## CNF 規模（base，不含累積 blocking）
| 方法 | Placements | Base 子句數 |
|------|------------|-------------|
| baseline | 6,120 | 44,201,651 |
| shape | 6,120 | 45,674,811 |

## 平均求解時間
| 方法 | 各次 (s) | 平均 (s) | 標準差 |
|------|----------|----------|--------|
| baseline | [20.714, 30.192, 30.817, 30.525, 31.265] | **28.703** | 4.483 |
| shape | [22.021, 23.858, 28.364, 23.345, 21.355] | **23.788** | 2.747 |

- shape / baseline 平均加速：**≈ 1.21×**

## 驗證
- baseline: 5/5 解通過 no-touch
- shape: 5/5 解通過 no-touch

## 解的視覺化

v2 拼圖形狀與我提供的已知解見
[`../../docs/images/v1-v2-puzzles.svg`](../../docs/images/v1-v2-puzzles.svg)；
本實驗 5 次枚舉的逐解資料保存在 `enum_v2/results.json`。
