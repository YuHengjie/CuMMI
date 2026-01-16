# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from matplotlib import font_manager as fm
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)


# %%
df_single = pd.read_csv("tabular_feature_single.csv")  
df_single

# %%
nano_paras = df_single['Feature'][1:15].values.tolist()
nano_paras

# %%
protein_paras = df_single['Feature'][15:18].values.tolist()
protein_paras

# %%
incubation_paras = df_single['Feature'][18:24].values.tolist()
incubation_paras

# %%
separation_paras = df_single['Feature'][24:29].values.tolist()
separation_paras

# %%
proteomic_paras = df_single['Feature'][29:37].values.tolist()
proteomic_paras

# %%
research_paras = [df_single['Feature'][37]]
research_paras

# %%
# 构造字典结构
feature_config = {
    "Nanomaterials": nano_paras,
    "Protein": protein_paras,
    "Incubation": incubation_paras,
    "Separation": separation_paras,
    "Proteomic": proteomic_paras,
    "Research purpose": research_paras
}

# 转换为 JSON 格式字符串 (indent=4 让输出更易读)
json_output = json.dumps(feature_config, indent=4, ensure_ascii=False)

# 打印查看
print(json_output)

# 如果需要保存到文件
with open('feature_config.json', 'w', encoding='utf-8') as f:
    json.dump(feature_config, f, indent=4, ensure_ascii=False)


# %%
group_order_pie = [
    "Nanomaterials",
    "Protein",
    "Incubation",
    "Proteomic",
    "Research purpose",
    "Separation",
]

group_colors_pie = {
    "Nanomaterials": "#ECBF90",
    "Protein": "#EFA9A2",
    "Incubation": "#94DACE",
    "Research purpose": "#94BDD9",
    "Proteomic": "#D8D490",
    "Separation":"#AAA4E2" ,
}

# %%
feature_stats = {key: len(value) for key, value in feature_config.items()}
feature_stats

# %%
# ===== 按指定顺序组织数据 =====
labels = group_order_pie
sizes = [feature_stats[g] for g in group_order_pie]
colors = [group_colors_pie[g] for g in group_order_pie]

fig, ax = plt.subplots(figsize=(2, 2))

def autopct_count(pct):
    total = sum(sizes)
    count = int(round(pct * total / 100.0))
    return f"{count}" if count > 0 else ""

wedges, texts, autotexts = ax.pie(
    sizes,
    colors=colors,
    startangle=180,
    counterclock=False,
    autopct=autopct_count,
    pctdistance=0.8,
    wedgeprops=dict(edgecolor='white', linewidth=1.2)
)

# 字体统一
for t in autotexts:
    t.set_fontproperties(prop)
    t.set_fontsize(12)
    t.set_color('black')

ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(
    "feature_group_distribution_pie.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()

# %%
# 提取第一行的 F1 作为基准值
baseline_f1 = df_single.loc[0, 'F1']

# 创建一个新的 DataFrame 来存储结果
f1_diff_df = pd.DataFrame({
    'Feature': df_single['Feature'][1:],  # 跳过第一行（Baseline）
    'F1 diff fill': baseline_f1 - df_single['F1'][1:] # 计算差值
})
f1_diff_df = f1_diff_df.sort_values(by='F1 diff fill', ascending=False)
f1_diff_df = f1_diff_df.reset_index(drop=True)
f1_diff_df

# %%
f1_diff_df.to_csv("f1_diff_single.csv", index=False)

# %%
f1_diff_df['Feature'] = f1_diff_df['Feature'].str.replace('℃', '°C')
f1_diff_df

# %%
df_plot = f1_diff_df.copy()
group_order = [ "Nanomaterials", "Protein", "Incubation", "Research purpose", "Proteomic", "Separation" ]

group_colors = { "Nanomaterials": "#EE9D40", 
                "Protein": "#F17466", 
                "Incubation": "#4CCCB7",
                "Research purpose": "#519ACD", 
                "Proteomic": "#C1BA30", 
                "Separation":"#7B72E2" , }
 
feature_config = {
    key: [item.replace('℃', '°C') for item in value] 
    for key, value in feature_config.items()
}

# Feature -> Group 映射
feature_to_group = {}
for g, feats in feature_config.items():
    for f in feats:
        feature_to_group[f] = g

# 为 df_plot 中每个 feature 分配颜色
colors = [
    group_colors.get(feature_to_group.get(f), "gray")
    for f in df_plot["Feature"]
]
colors

# %%

fig, ax = plt.subplots(figsize=(12, 4.5))

x_pos = np.arange(len(df_plot))

# ===== 1. 画“棒子”（从 0 到数值的线）=====
ax.vlines(
    x=x_pos,
    ymin=0,
    ymax=df_plot['F1 diff fill'],
    color='lightgray',
    linewidth=1.5,
    zorder=1
)

# ===== 2. 画“糖”（圆点）=====
# ===== 外圈：彩色描边 =====
ax.scatter(
    x_pos,
    df_plot['F1 diff fill'],
    s=140,
    facecolors='none',
    edgecolors=colors,
    linewidths=2.0,
    zorder=3
)

# ===== 中圈：白色实心（隔离背景）=====
ax.scatter(
    x_pos,
    df_plot['F1 diff fill'],
    s=90,
    facecolors='white',
    edgecolors='none',
    zorder=4
)

# ===== 内点：原本的彩色点 =====
ax.scatter(
    x_pos,
    df_plot['F1 diff fill'],
    s=45,
    color=colors,
    zorder=5
)

# ===== 3. 坐标轴刻度 =====
ax.set_xticks(x_pos)
ax.set_xticklabels(
    df_plot['Feature'],
    rotation=45,
    ha='right',
    fontproperties=prop,
    fontsize=12
)

ax.set_ylabel(
    'ΔF1 after feature ablation',
    fontproperties=prop,
    fontsize=12
)

# ===== 4. 美化 =====
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', length=4)


for label in ax.get_yticklabels():
    label.set_fontproperties(prop)

ax.set_ylim(0, 0.105)
ax.set_xlim(-0.5, len(df_plot)-0.5)
handles = []
labels = []

for g in group_order:
    c = group_colors[g]

    # 外圈（彩色描边）
    h_outer = Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markersize=11,
        markerfacecolor='none',
        markeredgecolor=c,
        markeredgewidth=2.0
    )

    # 中圈（白色实心）
    h_middle = Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markersize=8,
        markerfacecolor='white',
        markeredgecolor='white'
    )

    # 内点（彩色实心）
    h_inner = Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markersize=6,
        markerfacecolor=c,
        markeredgecolor=c
    )

    handles.append((h_outer, h_middle, h_inner))
    labels.append(g)

leg = ax.legend(
    handles=handles,
    labels=labels,
    handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},  # ⭐关键
    frameon=False,
    ncol=2,
    loc='upper right',            
    bbox_to_anchor=(0.5, 0.9),   
    prop=prop,
    fontsize=11,
    handletextpad=0.6,
    labelspacing=0.8,
    borderaxespad=0.0
)

plt.tight_layout()
plt.savefig(
    'F1_single_feature_importance.png',
    dpi=600,
    bbox_inches='tight'
)
plt.show()

# %%
