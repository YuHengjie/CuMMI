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
df_f1_two = pd.read_csv("tabular_module_two.csv",index_col=0)  
df_f1_two

# %%
df_f1_single = pd.read_csv("tabular_feature_module.csv")  
df_f1_single


# %% 添加对角线
'''
for feature in df_f1_two.columns:
    df_f1_two.loc[feature, feature] = df_f1_single.loc[df_f1_single['Feature'] == feature, 'F1'].values[0]
df_f1_two
'''

# %%
df_f1_baseline = df_f1_single.loc[0, 'F1']
df_f1_baseline

# %%
# 使用 df_f1_two.copy() 避免修改原始数据
df_f1_two_diff = df_f1_two.copy()

# 对每个元素进行操作：只对非 NaN 的值做减法
df_f1_two_diff = df_f1_two_diff.apply(lambda col: col.apply(lambda x: df_f1_baseline - x if pd.notna(x) else x))
df_f1_two_diff_T = df_f1_two_diff.T
df_f1_two_diff_T

# %%
# Find rows that are not all NaN
non_nan_rows = np.any(~np.isnan(df_f1_two_diff_T), axis=1)
df_f1_two_diff_T_filtered = df_f1_two_diff_T[non_nan_rows]

# Find columns that are not all NaN
non_nan_cols = np.any(~np.isnan(df_f1_two_diff_T_filtered), axis=0)
df_f1_two_diff_T_filtered = df_f1_two_diff_T_filtered.loc[:, non_nan_cols]
df_f1_two_diff_T_filtered

# %%
df = df_f1_two_diff_T_filtered
data = df.values
rows, cols = data.shape

fig, ax = plt.subplots(figsize=(4, 3))

# ===== 1) 只给“非 NaN”格子画灰色边框（内部白色）=====
mask = ~np.isnan(data)

for i in range(rows):
    for j in range(cols):
        if mask[i, j]:
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor='none',          # 内部保持白色
                edgecolor='lightgray',     # 灰色边框
                linewidth=0.8,
                zorder=0,
                clip_on=False              # 关键：避免边界裁剪导致最右/最下边框不显示
            )
            ax.add_patch(rect)

# ===== 2) 只绘制非 NaN 的圆点 =====
x, y = np.meshgrid(np.arange(cols), np.arange(rows))
x_plot = x[mask]
y_plot = y[mask]
values = data[mask]

# 圆大小映射（可调）
size_scale = 800
sizes = np.abs(values) * size_scale

# 颜色映射
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sc = ax.scatter(
    x_plot, y_plot,
    s=sizes,
    c=values,
    cmap=cmap,
    edgecolors='black',
    linewidths=0.5,
    zorder=2,
    clip_on=False
)

# ===== 3) 坐标轴刻度与字体 =====
ax.set_xticks(np.arange(cols))
ax.set_yticks(np.arange(rows))

ax.set_xticklabels(
    df.columns,
    rotation=45, ha='right',
    fontproperties=prop, fontsize=12
)
ax.set_yticklabels(
    df.index,
    fontproperties=prop, fontsize=12
)

# 与 seaborn heatmap 一致：第0行在最上面
ax.invert_yaxis()

# ===== 4) 关键：坐标范围稍微放大，保证边框线不被“贴边吞掉”=====
eps = 0.05
ax.set_xlim(-0.5 - eps, cols - 0.5 + eps)
ax.set_ylim(rows - 0.5 + eps, -0.5 - eps)

# 去掉刻度线
ax.tick_params(length=0)

# ===== 5) 只保留 左 + 下 外边框 =====
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)


# ===== 5) colorbar（无 label，只改字体）=====
cbar = plt.colorbar(sc, shrink=0.6, pad=-0.1,fraction=0.1, anchor=(0.0, 0.7))
cbar.ax.tick_params(labelsize=12)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(prop)

plt.tight_layout()
plt.savefig('Ablation_importance_binary_combined_module_bubble.png',
            dpi=600, bbox_inches='tight')
plt.show()

# %%
# 提取第一行的 F1 作为基准值
baseline_f1 = df_f1_single.loc[0, 'F1']

# 创建一个新的 DataFrame 来存储结果
f1_diff_hybrid_single = pd.DataFrame({
    'Feature': df_f1_single['Feature'][1:],  # 跳过第一行（Baseline）
    'F1 diff hybrid': baseline_f1 - df_f1_single['F1'][1:] # 计算差值
})
f1_diff_hybrid_single = f1_diff_hybrid_single.sort_values(by='F1 diff hybrid', ascending=False)
f1_diff_hybrid_single = f1_diff_hybrid_single.reset_index(drop=True)
f1_diff_hybrid_single

# %%
df_f1_two_inter = df_f1_two_diff.copy()
for feature1 in df_f1_two_diff.index:
    for feature2 in df_f1_two_diff.columns:
       if not pd.isna(df_f1_two_diff.loc[feature1, feature2]):
           df_f1_two_inter.loc[feature1, feature2] = df_f1_two_inter.loc[feature1, feature2] - f1_diff_hybrid_single.loc[f1_diff_hybrid_single['Feature'] == feature1, 'F1 diff hybrid'].values[0] - f1_diff_hybrid_single.loc[f1_diff_hybrid_single['Feature'] == feature2, 'F1 diff hybrid'].values[0]
df_f1_two_inter

# %%
df_f1_two_inter_T = df_f1_two_inter.T
df_f1_two_inter_T

# %%
# Find rows that are not all NaN
non_nan_rows = np.any(~np.isnan(df_f1_two_inter_T), axis=1)
df_f1_two_inter_T_filtered = df_f1_two_inter_T[non_nan_rows]

# Find columns that are not all NaN
non_nan_cols = np.any(~np.isnan(df_f1_two_inter_T_filtered), axis=0)
df_f1_two_inter_T_filtered = df_f1_two_inter_T_filtered.loc[:, non_nan_cols]
df_f1_two_inter_T_filtered


# %%
df_f1_two_inter_T_filtered.to_csv('df_f1_two_inter_T_filtered.csv',)

# %%
df = df_f1_two_inter_T_filtered
data = df.values
rows, cols = data.shape

fig, ax = plt.subplots(figsize=(4, 3))

# ===== 1) 只给“非 NaN”格子画灰色边框（内部白色）=====
mask = ~np.isnan(data)

for i in range(rows):
    for j in range(cols):
        if mask[i, j]:
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor='none',          # 内部保持白色
                edgecolor='lightgray',     # 灰色边框
                linewidth=0.8,
                zorder=0,
                clip_on=False              # 关键：避免边界裁剪导致最右/最下边框不显示
            )
            ax.add_patch(rect)

# ===== 2) 只绘制非 NaN 的圆点 =====
x, y = np.meshgrid(np.arange(cols), np.arange(rows))
x_plot = x[mask]
y_plot = y[mask]
values = data[mask]

# 圆大小映射（可调）
size_scale = 2500
sizes = np.abs(values) * size_scale

# 颜色映射
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sc = ax.scatter(
    x_plot, y_plot,
    s=sizes,
    c=values,
    cmap=cmap,
    edgecolors='black',
    linewidths=0.5,
    zorder=2,
    clip_on=False
)

# ===== 3) 坐标轴刻度与字体 =====
ax.set_xticks(np.arange(cols))
ax.set_yticks(np.arange(rows))

ax.set_xticklabels(
    df.columns,
    rotation=45, ha='right',
    fontproperties=prop, fontsize=12
)
ax.set_yticklabels(
    df.index,
    fontproperties=prop, fontsize=12
)

# 与 seaborn heatmap 一致：第0行在最上面
ax.invert_yaxis()

# ===== 4) 关键：坐标范围稍微放大，保证边框线不被“贴边吞掉”=====
eps = 0.05
ax.set_xlim(-0.5 - eps, cols - 0.5 + eps)
ax.set_ylim(rows - 0.5 + eps, -0.5 - eps)

# 去掉刻度线
ax.tick_params(length=0)

# ===== 5) 只保留 左 + 下 外边框 =====
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)


# ===== 5) colorbar（无 label，只改字体）=====
cbar = plt.colorbar(sc, shrink=0.6, pad=-0.1, fraction=0.1, anchor=(0.0, 0.7))
cbar.ax.tick_params(labelsize=12)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(prop)
    
plt.rcParams["axes.unicode_minus"] = False

plt.tight_layout()
plt.savefig('Ablation_importance_binary_inter_module_bubble.png',
            dpi=600, bbox_inches='tight')
plt.show()

# %%