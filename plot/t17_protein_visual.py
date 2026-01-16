# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms

# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'

# %%
df_fill = pd.read_csv("dataset_curated_reg.csv",keep_default_na=False, na_values=[''])
df_fill

# %%
df_non_fill = pd.read_csv("dataset_curated_reg_non_fill.csv",keep_default_na=False, na_values=[''])
df_non_fill

# %%
# 合并df
df_combined = pd.concat([df_fill, df_non_fill], ignore_index=True)
df_combined = df_combined.reset_index(drop=True)
df_combined

# %%
# 获取Accession列的独特值并统计个数
accession_counts = df_combined['Accession'].value_counts()
print(accession_counts)

# %%
# 获取前 m 个最常见的 Accession 值
top_accessions = accession_counts.head(34)
y_positions = range(len(top_accessions))

# 创建水平棒棒图
plt.figure(figsize=(2.5, 8.5))  # 注意调整图形大小，宽度大于高度
ax = plt.gca()  # 获取当前轴对象

# 绘制水平的线段和圆点
plt.hlines(y_positions, [0] * len(y_positions), top_accessions.values, 
           color='#3DA8DE', linestyle='--', linewidth=1)
plt.plot(top_accessions.values, y_positions, "o", color='#3DA8DE')

plt.ylabel('Accession', fontsize=14)
plt.xlabel('Sample count', fontsize=14)
# plt.title('Top 100 Most Frequent Accessions')

# 将 x 轴移动到顶部
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
# 获取 x 轴的刻度标签并设置字体大小
xticklabels = ax.get_xticklabels()
for label in xticklabels:
    label.set_fontsize(13) 

# 设置 y 轴标签
ax.set_yticks(y_positions)
ax.set_yticklabels(top_accessions.index, fontsize=13)

ax.set_ylim(-1, len(y_positions))

ax.set_xlim(top_accessions.values.min() * 0.95, 
            top_accessions.values.max() * 1.05)

ax.spines['right'].set_visible(False) 
ax.spines['bottom'].set_visible(False) 

plt.gca().invert_yaxis()  # 反转 y 轴，使得频率最高的在最上面
plt.tight_layout()

plt.savefig('data_visualize_y_pro_100.png', dpi=300,
            bbox_inches='tight', facecolor='none',
            transparent=True)

# %%
# 获取前 n 个最常见的 Accession 值
num_plot = len(accession_counts)
top_accessions = accession_counts.head(num_plot)

# 获取 Accession 值和对应的计数
accession_values = top_accessions.index
counts = top_accessions.values

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 3))  # 正确创建 fig 和 ax

# 绘制折线图并填充
x_positions = range(len(accession_values))
ax.plot(x_positions, counts, color='#0E6CB1', linewidth=1)
ax.fill_between(x_positions, counts, 0, color='#3DA8DE', alpha=0.4)

# 设置 x 轴刻度
xticks = np.arange(0, num_plot, 1000)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, rotation=90, fontsize=12)

# 手动偏移 tick labels 向右
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')
    trans = label.get_transform()
    offset = mtransforms.ScaledTranslation(6.3/72, 0, fig.dpi_scale_trans)  # 向右移动10pt
    label.set_transform(trans + offset)

yticks = np.arange(0, 4001, 1000)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=12)

# 旋转 y 轴刻度标签
for label in ax.get_yticklabels():
    label.set_rotation(90)
    label.set_verticalalignment('center')  # 可选：让文字居中对齐


# 设置标签
ax.set_xlabel('Accession index', rotation=180, fontsize=13)
ax.set_ylabel('Sample count', fontsize=13)

# 美化边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置轴范围
ax.set_ylim(0, counts.max() * 1.05)
ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('data_visualize_y_pro_33736.png', dpi=300,
            bbox_inches='tight', facecolor='none',
            transparent=True)

# %%
