# 實驗產物

這個 repository 保留程式、結果摘要、精簡 JSON 與圖表。體積較大、可由程式重建的原始產物另存於 GitHub Release
[`research-artifacts-v1`](https://github.com/Michaelwu128/new_puzzle_2026/releases/tag/research-artifacts-v1)，避免一般 clone 下載大量中間資料。

## Release 檔案

| 檔案 | 內容 | 原始檔數 | SHA-256 |
|---|---|---:|---|
| `dead-pocket-oracle-labels-v1-v6.tar.gz` | v1、v2、v4、v5、v6 的完整 dead-pocket oracle 標記（`full_dead_v*.json`） | 5 | `523c9811dd0d444d4e83d01fd96ddd151bcb99fe0da9fa08611e0857d3fc337e` |
| `placement-metadata.tar.gz` | dataset / benchmark metadata（`*_meta.json`；多數包含 placement-to-variable 映射） | 32 | `ff447577839720521d209c97a7660eb5eee7254329c78c41f6156d9414de03a0` |
| `v6-break-even-solutions-k100.tar.gz` | v6 baseline、shape、learned_shape 各 100 個 Kissat 解答 | 300 | `1a744906c0e48e6b5ec0139ce19063dc6955c801a1340c1328630ac3ffe7507c` |

下載後可驗證：

```bash
sha256sum -c SHA256SUMS
```

## 為什麼不直接放在 Git

這些檔案是可重建的實驗中間產物，合計超過 200 MiB，而且 placement metadata 在不同 benchmark
間包含大量重複內容。repository 內的 `RESULTS.md`、benchmark JSON 與圖表已保留重現設定和摘要；
需要逐筆 oracle 標記、placement 映射或 SAT assignment 時，再下載 Release 即可。

## 重建方式

- `full_dead_v*.json`：使用 `encoding/build_dead_pocket_dataset.py` 與對應 puzzle 重新執行 oracle 掃描。
- `dataset_meta.json`：由 `encoding/build_dead_pocket_dataset.py` 輸出；其他 `*_meta.json`
  由 `encoding/generate_cnf.py` 或 benchmark 腳本產生 CNF 時一併輸出。
- v6 break-even 解答：使用 `encoding/run_break_even_benchmark.py --puzzle v6 --k-max 100`。

完整參數與實驗協議見
[`baselines/learned_shape/EXPERIMENTS_DETAILED.md`](baselines/learned_shape/EXPERIMENTS_DETAILED.md)。

## 公開版匿名化

公開 repository 中保留的 Kissat 文字輸出已將個人伺服器絕對路徑改為 repo-relative path，
並將 solver banner 的主機名稱改為 `<host>`；SAT status、assignment、時間與統計數據未變。
Release archives 是匿名化前從原始產物複製的封存檔；整理前的完整 Git tree 另保存在本機
`git bundle`。
