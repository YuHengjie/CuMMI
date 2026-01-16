# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

# %%
train_all_np = '../train_basic_10/output/stage_12345/roc_internal_stage_5.npy'
train_protein_np = '../train_basic_only_protein/output/stage_12345/roc_internal_stage_5.npy'
train_text_np = '../train_basic_only_text/output/stage_12345/roc_internal_stage_5.npy'

roc_all = np.load(train_all_np, allow_pickle=True).item()
roc_protein = np.load(train_protein_np, allow_pickle=True).item()
roc_text = np.load(train_text_np, allow_pickle=True).item()

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))

def plot_roc(ax, roc, label, color, linestyle):
    fpr = roc["fpr"]
    tpr = roc["tpr"]
    auc = roc["auc"]

    ax.plot(
        fpr, tpr,
        color=color,
        lw=1.5,
        linestyle=linestyle,
        label=f"{label} (AUROC = {auc:.2f})"
    )


plot_roc(ax, roc_all,     "All modalities", "#F94141", "-")    # 实线
plot_roc(ax, roc_protein, "Protein only",   "#37AB78", "--")   # 虚线
plot_roc(ax, roc_text,    "Text only",      "#589FF3", "-.")   # 点划线

# 随机参考线
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)


ax.set_xlabel("False positive rate", fontproperties=prop, fontsize=12)
ax.set_ylabel("True positive rate", fontproperties=prop, fontsize=12)


# 刻度字体
ax.tick_params(axis="both", labelsize=10)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontproperties(prop)

# 图例字体
legend = ax.legend(
    frameon=False,
    loc="lower right",
    fontsize=10,
    prop=prop
)

# ======================
# 4. Nature 风格边框
# ======================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

ax.set_xlim(-0.05, 1.02)
ax.set_ylim(0, 1.01)

plt.tight_layout()
plt.savefig(
    "ROC_internal_all_vs_protein_vs_text.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()



# %%
# %%
train_all_np = '../train_basic_10/output/stage_12345/pr_internal_stage_5.npy'
train_protein_np = '../train_basic_only_protein/output/stage_12345/pr_internal_stage_5.npy'
train_text_np = '../train_basic_only_text/output/stage_12345/pr_internal_stage_5.npy'

pr_all = np.load(train_all_np, allow_pickle=True).item()
pr_protein = np.load(train_protein_np, allow_pickle=True).item()
pr_text = np.load(train_text_np, allow_pickle=True).item()

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))

def plot_pr(ax, pr, label, color, linestyle):
    recall = pr["recall"]
    precision = pr["precision"]
    ap = pr["ap"]

    ax.plot(
        recall, precision,
        color=color,
        lw=1.5,
        linestyle=linestyle,
        label=f"{label} (AUPRC = {ap:.2f})"
    )

plot_pr(ax, pr_all,     "All modalities", "#F94141", "-")
plot_pr(ax, pr_protein, "Protein only",   "#37AB78", "--")
plot_pr(ax, pr_text,    "Text only",      "#589FF3", "-.")

# baseline：正类比例（随机分类器）
pos_rate = pr_all["y_true"].mean()
ax.hlines(
    pos_rate, xmin=0, xmax=1,
    colors="gray", linestyles="--", linewidth=1,
    label=f"Baseline = {pos_rate:.2f}"
)

# 轴标签
ax.set_xlabel("Recall", fontproperties=prop, fontsize=12)
ax.set_ylabel("Precision", fontproperties=prop, fontsize=12)

# 刻度字体
ax.tick_params(axis="both", labelsize=10)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontproperties(prop)

# 图例
legend = ax.legend(
    frameon=False,
    loc="lower left",
    fontsize=10,
    prop=prop
)

# Nature 风格边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

ax.set_xlim(0, 1.02)
ax.set_ylim(0.28, 1.02)

plt.tight_layout()
plt.savefig(
    "PR_internal_all_vs_protein_vs_text.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

# %%