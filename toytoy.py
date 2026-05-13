import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# =========================
# 日本語フォント設定
# =========================#
font_candidates = ['Apple SD Gothic Neo', 'Hiragino Sans GB', 'AppleGothic']
for font_name in font_candidates:
    if any(font_name == f.name for f in fm.fontManager.ttflist):
        mpl.rcParams['font.family'] = font_name
        break

mpl.rcParams['axes.unicode_minus'] = False

# =========================
# 設定
# =========================
m = 120          # 仮説数
alpha = 0.10     # 有意水準
np.random.seed(7)

# =========================
# x軸: 仮説の順位
# =========================
i = np.arange(1, m + 1)

# =========================
# 従来法(BY)の閾値
# =========================
c_m = np.sum(1 / np.arange(1, m + 1))
by_thr = alpha * i / (m * c_m)

# =========================
# 提案法の閾値
#  - BYより緩い
#  - 相関構造を反映した adaptive threshold の概念図
# =========================
# 先頭ほど少し厚めに閾値を与える
adaptive_factor = 1.8 * (1.0 + 0.35 * np.exp(-(i - 1) / 28))
prop_thr = by_thr * adaptive_factor

# 単調増加を保証
prop_thr = np.maximum.accumulate(prop_thr)

# =========================
# 並べ替えた p値（概念例）
#  - 最初の数個は BY でも棄却
#  - 次のいくつかは提案法なら棄却
#  - 残りは棄却されない
# =========================
p = np.zeros(m)

n_both = 5          # BYでも棄却
n_prop_only = 18    # 提案法のみで棄却

# BYでも棄却される点
p[:n_both] = by_thr[:n_both] * np.linspace(0.45, 0.75, n_both)

# 提案法のみで棄却される点
p[n_both:n_both + n_prop_only] = (
    by_thr[n_both:n_both + n_prop_only] * 0.55
    + prop_thr[n_both:n_both + n_prop_only] * 0.45
)

# 非発見の点
rest = m - (n_both + n_prop_only)
p[n_both + n_prop_only:] = (
    prop_thr[n_both + n_prop_only:] + np.linspace(0.002, 0.08, rest)
)

# 念のため単調増加にする
p = np.maximum.accumulate(p)

# =========================
# カテゴリ分け
# =========================
rej_by = p <= by_thr
rej_prop = p <= prop_thr

both = rej_by
prop_only = rej_prop & (~rej_by)
non = ~rej_prop

# =========================
# 作図
# =========================
fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=200)

# 点
ax.scatter(i[non], p[non],
           facecolors='white', edgecolors='black',
           s=34, marker='o', linewidths=0.9,
           label='非棄却', zorder=3)

ax.scatter(i[prop_only], p[prop_only],
           facecolors='black', edgecolors='black',
           s=34, marker='o', linewidths=0.9,
           label='提案法のみ棄却', zorder=4)

ax.scatter(i[both], p[both],
           facecolors='black', edgecolors='black',
           s=52, marker='^', linewidths=0.9,
           label='BYでも棄却', zorder=5)

# 閾値線
ax.plot(i, by_thr, linestyle='--', linewidth=1.8, color='black',
        label='BY', zorder=2)

ax.plot(i, prop_thr, linestyle='-', linewidth=1.8, color='black',
        label='提案法（相関構造を反映）', zorder=2)

# 軸ラベル
ax.set_xlabel('仮説の順位', fontsize=13)
ax.set_ylabel('並べ替えた $p$ 値', fontsize=13)

# タイトル
ax.set_title('相関構造を考慮した多重検定', fontsize=16, pad=12)

# 線の注記
ax.text(m * 0.82, by_thr[int(m * 0.82)] + 0.006, 'BY', fontsize=12)
ax.text(m * 0.82, prop_thr[int(m * 0.82)] + 0.006, '提案法', fontsize=12)

# 棄却数の注記
n_by = np.sum(rej_by)
n_prop = np.sum(rej_prop)

ax.text(0.03, 0.95,
        f'BYの棄却数: {n_by}\n提案法の棄却数: {n_prop}',
        transform=ax.transAxes,
        va='top', ha='left', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

# 凡例
ax.legend(frameon=False, fontsize=11, loc='lower right')

# 見た目調整
ax.set_xlim(0, m + 3)
ax.set_ylim(0, min(1.0, p.max() + 0.04))
ax.tick_params(axis='both', labelsize=11)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.show()

# 保存したい場合
# plt.savefig("multiple_testing_conceptual_plot.png", dpi=300, bbox_inches="tight")