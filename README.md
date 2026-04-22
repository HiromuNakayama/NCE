# NCE Toy Example

1次元ガウス分布で **Noise-Contrastive Estimation (NCE)** を最小実装したサンプルです。

## 実行

```bash
python3 nce_toy.py
```

学習ログと最終結果として、以下を表示します。

- 真の平均 `mu_true`
- MLE による平均推定 `mu_mle`
- NCE による平均推定 `mu_nce`
- 真の `logZ`
- NCE が推定した `c`（`logZ` に対応）

## ポイント

- NCE は、データ vs ノイズの2値分類に帰着して学習できます。
- この例では未正規化モデル `exp(log p_tilde - c)` の `mu` と `c` を同時に学習します。
- Toy なので可読性を優先し、標準ライブラリのみで勾配を手計算しています。
