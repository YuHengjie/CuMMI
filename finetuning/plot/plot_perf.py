# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

# %%
train_nano_scratch_np = '../train_nano/output/from_scratch/eval_summary_external.csv'
train_nano_ft_np = '../train_nano/output/ft/eval_summary_external.csv'

df_nano_scratch = pd.read_csv(train_nano_scratch_np)
df_nano_scratch

# %%
df_nano_ft = pd.read_csv(train_nano_ft_np)
df_nano_ft

# %%
# 确保按 framework 排序（10,20,...,100）
df_nano_scratch   = df_nano_scratch.sort_values('framework')
df_nano_ft = df_nano_ft.sort_values('framework')
df_nano_scratch

# %%
mean_gain = (
    df_nano_ft['AUROC'].values
    - df_nano_scratch['AUROC'].values
).mean()

mean_gain


# %%
fig, ax = plt.subplots(figsize=(6.5, 3))

# 第一条线：nano scratch
ax.plot(
    df_nano_scratch['framework'],
    df_nano_scratch['AUROC'],
    marker='o',
    linewidth=2,
    markersize=6,
    color="#75A0AF",
    label='From scratch'
)

# 第二条线：nano fine-tune
ax.plot(
    df_nano_ft['framework'],
    df_nano_ft['AUROC'],
    marker='s',
    linewidth=2,
    markersize=6,
    color='#DF8389',
    label='Finetuning'
)

ax.margins(y=0)  # 取消y方向自动留白
ax.set_ylim((0.88,0.93))

# ---- 辅助线：从 df_nano_ft 第一个点的 AUROC 水平连到 df_nano_scratch 曲线，并下落到 x 轴 ----

# 1) 取 df_nano_ft 第一个点
x0 = df_nano_ft['framework'].iloc[0]
y0 = df_nano_ft['AUROC'].iloc[0]

# 2) 在 df_nano_scratch 上找 AUROC=y0 对应的 x（线性插值）
xs = df_nano_scratch['framework'].to_numpy(dtype=float)
ys = df_nano_scratch['AUROC'].to_numpy(dtype=float)

# 先找 y0 落在哪个相邻区间（允许递增或递减）
idx = None
for i in range(len(ys) - 1):
    y1, y2 = ys[i], ys[i+1]
    if (y1 - y0) * (y2 - y0) <= 0 and (y1 != y2):
        idx = i
        break

if idx is None:
    print("y0 不在 df_nano_scratch 的 AUROC 范围内，无法插值求交点。")
else:
    x1, x2 = xs[idx], xs[idx+1]
    y1, y2 = ys[idx], ys[idx+1]

    # 线性插值：y0 = y1 + (y2-y1)*(x-x1)/(x2-x1)
    x_cross = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)

    # 3) 画辅助线
    # 水平线：从 df_nano_ft 第一个点 -> scratch 曲线交点
    ax.hlines(y=y0, xmin=x0, xmax=x_cross, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # 竖线：从交点 -> x轴（用当前y轴下限）
    y_bottom = ax.get_ylim()[0]
    ax.vlines(x=x_cross, ymin=y_bottom, ymax=y0, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # 交点做个标记（可选）
    ax.scatter(
        [x_cross], [y0],
        s=40,                      # 稍微大一点更清楚
        color='#DF8389',      # 边缘颜色
        linewidths=1.5,
        zorder=5,
        alpha=0.7,
    )
    # 标注 x_cross（可选）
    ax.annotate(f"{x_cross:.1f}",
                xy=(x_cross, y_bottom),
                xytext=(15, 10),
                textcoords="offset points",
                ha='center', va='top', color='#D54952',
                fontproperties=prop, fontsize=10)

ax.text(
    0.75, 0.25,
    f"Mean AUROC gain = +{mean_gain:.3f}\n(at the same training data proportion) ",
    transform=ax.transAxes,
    ha='center', va='top',
    fontproperties=prop,
    color = '#D54952',
    fontsize=10
)

# from scratch 第一个点
x_s = df_nano_scratch['framework'].iloc[0]
y_s = df_nano_scratch['AUROC'].iloc[0]

# finetuning 第一个点
x_f = df_nano_ft['framework'].iloc[0]
y_f = df_nano_ft['AUROC'].iloc[0]

ax.annotate(
    "",                      # 不写文字，只画箭头
    xy=(x_f, y_f),           # 箭头终点（finetuning）
    xytext=(x_s, y_s),       # 箭头起点（scratch）
    arrowprops=dict(
        arrowstyle="->",
        color="#DF8389",
        linewidth=1.5,
        alpha=0.8,
        shrinkA=2,
        shrinkB=2
    ),
    zorder=4
)

# 箭头中点
x_mid = (x_s + x_f) / 2
y_mid = (y_s + y_f) / 2

ax.text(
    x_mid+0.5, y_mid,
    f"+{(y_f - y_s):.3f}",
    color="#D54952",
    fontproperties=prop,
    fontsize=10,
    ha='left',
    va='center',
    zorder=5
)

# 坐标轴标签
ax.set_xlabel("Fraction of training data (%)", fontproperties=prop, fontsize=12)
ax.set_ylabel("AUROC", fontproperties=prop, fontsize=12)

# x 轴刻度：10 到 100，间隔 10
ax.set_xticks(np.arange(10, 101, 10))
for tick in ax.get_xticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# y 轴刻度
for tick in ax.get_yticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# 图例
leg = ax.legend(prop=prop, fontsize=11, frameon=False)

# 美化
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

plt.tight_layout()
plt.savefig(
    "Nano_scratch_ft.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()






# %%
# %%
train_protein_scratch_np = '../train_protein/output/from_scratch/eval_summary_external.csv'
train_protein_ft_np = '../train_protein/output/ft/eval_summary_external.csv'

df_protein_scratch = pd.read_csv(train_protein_scratch_np)
df_protein_scratch

# %%
df_protein_ft = pd.read_csv(train_protein_ft_np)
df_protein_ft

# %%
# 确保按 framework 排序（10,20,...,100）
df_protein_scratch   = df_protein_scratch.sort_values('framework')
df_protein_ft = df_protein_ft.sort_values('framework')
df_protein_scratch

# %%
mean_gain = (
    df_protein_ft['AUROC'].values
    - df_protein_scratch['AUROC'].values
).mean()

mean_gain

# %%
fig, ax = plt.subplots(figsize=(6.5, 3))

# 第一条线：protein scratch
ax.plot(
    df_protein_scratch['framework'],
    df_protein_scratch['AUROC'],
    marker='o',
    linewidth=2,
    markersize=6,
    color="#75A0AF",
    label='From scratch'
)

# 第二条线：protein fine-tune
ax.plot(
    df_protein_ft['framework'],
    df_protein_ft['AUROC'],
    marker='s',
    linewidth=2,
    markersize=6,
    color='#DF8389',
    label='Finetuning'
)

ax.margins(y=0)  # 取消y方向自动留白
ax.set_ylim((0.83,0.93))

# ---- 辅助线：从 df_protein_ft 第一个点的 AUROC 水平连到 df_protein_scratch 曲线，并下落到 x 轴 ----

# 1) 取 df_protein_ft 第一个点
x0 = df_protein_ft['framework'].iloc[0]
y0 = df_protein_ft['AUROC'].iloc[0]

# 2) 在 df_protein_scratch 上找 AUROC=y0 对应的 x（线性插值）
xs = df_protein_scratch['framework'].to_numpy(dtype=float)
ys = df_protein_scratch['AUROC'].to_numpy(dtype=float)

# 先找 y0 落在哪个相邻区间（允许递增或递减）
idx = None
for i in range(len(ys) - 1):
    y1, y2 = ys[i], ys[i+1]
    if (y1 - y0) * (y2 - y0) <= 0 and (y1 != y2):
        idx = i
        break

if idx is None:
    print("y0 不在 df_protein_scratch 的 AUROC 范围内，无法插值求交点。")
else:
    x1, x2 = xs[idx], xs[idx+1]
    y1, y2 = ys[idx], ys[idx+1]

    # 线性插值：y0 = y1 + (y2-y1)*(x-x1)/(x2-x1)
    x_cross = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)

    # 3) 画辅助线
    # 水平线：从 df_protein_ft 第一个点 -> scratch 曲线交点
    ax.hlines(y=y0, xmin=x0, xmax=x_cross, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # 竖线：从交点 -> x轴（用当前y轴下限）
    y_bottom = ax.get_ylim()[0]
    ax.vlines(x=x_cross, ymin=y_bottom, ymax=y0, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # 交点做个标记（可选）
    ax.scatter(
        [x_cross], [y0],
        s=40,                      # 稍微大一点更清楚
        color='#DF8389',      # 边缘颜色
        linewidths=1.5,
        zorder=5,
        alpha=0.7,
    )
    # 标注 x_cross（可选）
    ax.annotate(f"{x_cross:.1f}",
                xy=(x_cross, y_bottom),
                xytext=(15, 10),
                textcoords="offset points",
                ha='center', va='top', color='#D54952',
                fontproperties=prop, fontsize=10)

ax.text(
    0.75, 0.25,
    f"Mean AUROC gain = +{mean_gain:.3f}\n(at the same training data proportion) ",
    transform=ax.transAxes,
    ha='center', va='top',
    fontproperties=prop,
    color = '#D54952',
    fontsize=10
)

# from scratch 第一个点
x_s = df_protein_scratch['framework'].iloc[0]
y_s = df_protein_scratch['AUROC'].iloc[0]

# finetuning 第一个点
x_f = df_protein_ft['framework'].iloc[0]
y_f = df_protein_ft['AUROC'].iloc[0]

ax.annotate(
    "",                      # 不写文字，只画箭头
    xy=(x_f, y_f),           # 箭头终点（finetuning）
    xytext=(x_s, y_s),       # 箭头起点（scratch）
    arrowprops=dict(
        arrowstyle="->",
        color="#DF8389",
        linewidth=1.5,
        alpha=0.8,
        shrinkA=2,
        shrinkB=2
    ),
    zorder=4
)

# 箭头中点
x_mid = (x_s + x_f) / 2
y_mid = (y_s + y_f) / 2

ax.text(
    x_mid+0.5, y_mid,
    f"+{(y_f - y_s):.3f}",
    color="#D54952",
    fontproperties=prop,
    fontsize=10,
    ha='left',
    va='center',
    zorder=5
)


# 坐标轴标签
ax.set_xlabel("Fraction of training data (%)", fontproperties=prop, fontsize=12)
ax.set_ylabel("AUROC", fontproperties=prop, fontsize=12)

# x 轴刻度：10 到 100，间隔 10
ax.set_xticks(np.arange(10, 101, 10))
for tick in ax.get_xticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# y 轴刻度
for tick in ax.get_yticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# 图例
leg = ax.legend(prop=prop, fontsize=11, frameon=False)

# 美化
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

plt.tight_layout()
plt.savefig(
    "Protein_scratch_ft.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

# %%
