# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from matplotlib import font_manager as fm
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)


# %%
df_module = pd.read_csv("tabular_feature_module.csv")  
df_module

# %%
# 读取 JSON 文件
with open('../model_explain/feature_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
config

# %%
# 1. 获取所有 Key 并计算对应 List 的长度
feature_stats = {key: len(value) for key, value in config.items()}
feature_stats

# %%
# 提取第一行的 F1 作为基准值
baseline_f1 = df_module.loc[0, 'F1']

# 创建一个新的 DataFrame 来存储结果
f1_diff_df = pd.DataFrame({
    'Feature': df_module['Feature'][1:],  # 跳过第一行（Baseline）
    'F1 diff fill': baseline_f1 - df_module['F1'][1:] # 计算差值
})
f1_diff_df = f1_diff_df.sort_values(by='F1 diff fill', ascending=False)
f1_diff_df = f1_diff_df.reset_index(drop=True)
f1_diff_df

# %%
# 排序（从大到小，若你已排好可省略）
df_plot = f1_diff_df.sort_values("F1 diff fill", ascending=True)
# 加上数量与平均值
df_plot["Count"] = df_plot["Feature"].map(feature_stats)
df_plot["Avg diff"] = df_plot["F1 diff fill"] / df_plot["Count"]
df_plot

# %%
# y 位置（保证 bar 和 line 完全对齐）
y_pos = np.arange(len(df_plot))
y_pos

# %%
fig, ax = plt.subplots(figsize=(5, 2.2))

# y 位置
y_pos = np.arange(len(df_plot))

group_colors = { "Nanomaterials": "#EE9D40", 
                "Protein": "#F17466", 
                "Incubation": "#4CCCB7",
                "Research purpose": "#519ACD", 
                "Proteomic": "#C1BA30", 
                "Separation":"#7B72E2" , }

# 如果 Feature 不在 group_colors 里，用灰色兜底
colors = df_plot["Feature"].map(group_colors).fillna("gray")

# ===== 1️⃣ 画“棒子”（从 0 到数值的横线）=====
ax.hlines(
    y=y_pos,
    xmin=0,
    xmax=df_plot["F1 diff fill"],
    color="lightgray",
    linewidth=1.5,
    zorder=1
)

# ===== 2️⃣ 画“糖”（三层圆点）=====
# 外圈：彩色描边
ax.scatter(
    df_plot["F1 diff fill"],
    y_pos,
    s=140,
    facecolors="none",
    edgecolors=colors,
    linewidths=2.0,
    zorder=3
)

# 中圈：白色实心（隔离背景）
ax.scatter(
    df_plot["F1 diff fill"],
    y_pos,
    s=90,
    facecolors="white",
    edgecolors="none",
    zorder=4
)

# 内点：实心彩色
ax.scatter(
    df_plot["F1 diff fill"],
    y_pos,
    s=45,
    color=colors,
    zorder=5
)

# ===== 3️⃣ 坐标轴刻度 =====
ax.set_yticklabels([])
ax.tick_params(axis='y', left=False)  # 连y轴小刻度线也不要

# 在柱子右侧标注 feature_stats 数值
for i, (feature, value) in enumerate(
    zip(df_plot["Feature"], df_plot["F1 diff fill"])
):
    stat = feature_stats.get(feature, None)
    if stat is not None:
        ax.text(
            value + 0.008,   # 右侧轻微偏移（可根据数据大小微调）
            i,
            f"{stat}",
            va="center",
            ha="left",
            fontsize=12,
            fontproperties=prop,
            color="black"
        )

# 轴标签（解释性命名）
ax.set_xlabel(
    "ΔF1 after grouped-feature ablation",
    fontproperties=prop,
    fontsize=12,
    labelpad=8
)

# x刻度和xlabel移动到上方
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(axis="x", labelsize=13, top=True, bottom=False, labeltop=True, labelbottom=False)
for lab in ax.get_xticklabels():
    lab.set_fontproperties(prop)
    
# 字体
ax.set_yticklabels([], fontproperties=prop, fontsize=13)


ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Nature 风格：去掉多余边框
ax.spines["top"].set_color("black")
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_visible(False)

ax.spines["left"].set_linewidth(0.8)
ax.spines["top"].set_linewidth(0.8)


ax.set_ylim(-0.45, len(df_plot)-0.2)
ax.set_xlim(0.0, 0.28)

plt.tight_layout()
plt.savefig(
    "F1_group_importance_bar.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

# %%
f1_diff_df.to_csv("f1_diff_module.csv", index=False)

# %%
