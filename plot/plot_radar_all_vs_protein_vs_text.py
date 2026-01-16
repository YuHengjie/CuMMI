# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib import font_manager as fm
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

# %%
train_all_json = '../train_basic_10/output/stage_12345/eval_results.json'
train_protein_json = '../train_basic_only_protein/output/stage_12345/eval_results.json'
train_text_json = '../train_basic_only_text/output/stage_12345/eval_results.json'

# %%
metric_keys = ["AUROC", "AUPRC", "Precision", "Recall", "F1", "ACC"]

def safe_get_at_thr_metrics(path, fold_key="5"):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # fold key 兼容 str/int
    fold = d.get(fold_key, None)
    if fold is None:
        try:
            fold = d.get(int(fold_key), None)
        except Exception:
            fold = None

    internal = None if fold is None else fold.get("internal", None)
    m = None if internal is None else internal.get("at_thr", None)

    row = {k: np.nan for k in metric_keys}
    if isinstance(m, dict):
        for k in metric_keys:
            if k in m:
                row[k] = m[k]
    return row

rows = [
    {"model": "All modalities", **safe_get_at_thr_metrics(train_all_json, fold_key="5")},
    {"model": "Protein only", **safe_get_at_thr_metrics(train_protein_json, fold_key="5")},
    {"model": "Text only", **safe_get_at_thr_metrics(train_text_json, fold_key="5")},
]

df = pd.DataFrame(rows).set_index("model")
df

# %%
df = df.rename(columns={"ACC": "Accuracy"})
df

# %%
labels = ['Accuracy', 'Precision', 'Recall', 'AUPRC', 'AUROC']

stats_multi_modal = [df.loc["All modalities", l] for l in labels]
stats_pro_modal   = [df.loc["Protein only", l] for l in labels]
stats_nano_modal  = [df.loc["Text only", l] for l in labels]
stats_pro_modal

# %%
# 指标数量
num_metrics = len(labels)

# 计算每个轴的角度，并将 "Accuracy" 设置在正上方
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

# 找到需要的偏移量，使得第一个角度等于 π/2
offset = np.pi / 2 - angles[-1]

# 统一加上偏移量
angles = [(angle + offset) % (2 * np.pi) for angle in angles]

# 确保数据与角度的长度一致
stats_nano_modal += stats_nano_modal[:1]  # 使数据闭合
stats_pro_modal += stats_pro_modal[:1]  # 使数据闭合
stats_multi_modal += stats_multi_modal[:1]  # 使数据闭合

angles += angles[:1]  # 使角度闭合

# %%
# 创建雷达图
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

# 绘制两条模型线
ax.plot(angles, stats_multi_modal, marker='^', markersize=4,
        color='#6495ED', linewidth=1, linestyle='solid', label='All modalities')
ax.plot(angles, stats_pro_modal, marker='s', markersize=4,
        color='#59BAC3', linewidth=1, linestyle='solid', label='Protein only')
ax.plot(angles, stats_nano_modal, marker='o',  markersize=4,
        color='#E0817D', linewidth=1, linestyle='solid', label='Text only')

# 填充模型区域
ax.fill(angles, stats_multi_modal, color='#87CEFA', alpha=0.2)
ax.fill(angles, stats_pro_modal, color='#EBF8F8', alpha=0.7)
ax.fill(angles, stats_nano_modal, color='#F7F0E8', alpha=0.7)

# 设置轴标签
ax.set_xticks(angles[:-1])  # 不重复最后一个角度

ax.set_xticklabels(labels, fontproperties=prop)

# 获取标签对象列表
xticklabels = ax.get_xticklabels()
# 修改第一个标签的位置 (索引为 0)
xticklabels[0].set_position((0, 0.06)) # 调整 tangential_position

# 手动调整第二个和第五个标签的位置
for i, label in enumerate(ax.get_xticklabels()):
    if i == 0:  
        label.set_position((label.get_position()[0] + 0.1, label.get_position()[1] - 0.18))  # 右移、下移
    elif i == 3: 
        label.set_position((label.get_position()[0] + 0.1, label.get_position()[1] - 0.1))  # 右移、下移

ax.set_yticks(np.linspace(0.4, 1, num=4))  # 设置y轴为5个刻度（包括最小值和最大值）
ax.set_ylim(0.4, 1)  # 设置雷达图的范围为 [0.4, 1]

# 关键：设置多边形样式（关闭圆形网格）
ax.yaxis.grid(False)  # 关闭圆形网格线
ax.xaxis.grid(True, linestyle='--',  linewidth=0.5, color='gray')  # 设置多边形网格线

# 设置 y 轴范围和刻度（去掉 1.0）
y_ticks = np.linspace(0.4, 1.0, num=4)[:-1]  # 刻度值为 [0.4, 0.6, 0.8]，去掉 1.0
ax.set_yticks(y_ticks,)
# 修改 y 轴刻度标签的字体大小
ax.set_yticklabels(y_ticks, fontproperties=prop, fontsize=8)  # 设置字体大小为 12

# 手动添加刻度线连接（多边形样式，仍然绘制到 1.0）
for value in np.linspace(0.4, 1.0, num=4):  # 绘制网格线到 1.0
    grid_line = [value] * (len(labels) + 1)  # 每个轴上的刻度值
    if value == 1.0:
        ax.plot(angles, grid_line, color='gray', linewidth=0.75, linestyle='-')
    else:
        ax.plot(angles, grid_line, color='gray', linewidth=0.5, linestyle='--')
    
# 设置刻度标签（只显示到 0.8）
ax.set_yticklabels([f'{val:.1f}' for val in y_ticks], color='gray')  # 显示刻度标签

# 关闭圆形网格线
ax.yaxis.grid(False)
ax.spines['polar'].set_visible(False)

# 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(0.3, 1.1), prop=prop, frameon=False)

plt.tight_layout()
plt.savefig('radar_modal compare_internal.png', dpi=600)  # dpi 参数控制图片分辨率
#plt.savefig('modal compare-binary.svg', )  # dpi 参数控制图片分辨率






# %%
train_internal_json = '../train_basic_10/output/stage_12345/eval_results.json'
train_external_json = '../train_date_10/output/stage_12345/eval_results.json'

# %%
metric_keys = ["AUROC", "AUPRC", "Precision", "Recall", "F1", "ACC"]

def safe_get_at_thr_metrics_external(path, fold_key="5"):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # fold key 兼容 str/int
    fold = d.get(fold_key, None)
    if fold is None:
        try:
            fold = d.get(int(fold_key), None)
        except Exception:
            fold = None

    internal = None if fold is None else fold.get("external", None)
    m = None if internal is None else internal.get("at_thr", None)

    row = {k: np.nan for k in metric_keys}
    if isinstance(m, dict):
        for k in metric_keys:
            if k in m:
                row[k] = m[k]
    return row

rows = [
    {"model": "Internal test set", **safe_get_at_thr_metrics(train_internal_json, fold_key="5")},
    {"model": "External test set", **safe_get_at_thr_metrics_external(train_external_json, fold_key="5")},
]

df = pd.DataFrame(rows).set_index("model")
df

# %%
df = df.rename(columns={"ACC": "Accuracy"})
df

# %%
labels = ['Accuracy', 'Precision', 'Recall', 'AUPRC', 'AUROC']

stats_internal_modal = [df.loc["Internal test set", l] for l in labels]
stats_external_modal = [df.loc["External test set", l] for l in labels]
stats_external_modal

# %%
stats_baseline_modal = [0.7] * 5
stats_baseline_modal

# %%
# 指标数量
num_metrics = len(labels)

# 计算每个轴的角度，并将 "Accuracy" 设置在正上方
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

# 找到需要的偏移量，使得第一个角度等于 π/2
offset = np.pi / 2 - angles[-1]

# 统一加上偏移量
angles = [(angle + offset) % (2 * np.pi) for angle in angles]

# 确保数据与角度的长度一致
stats_internal_modal += stats_internal_modal[:1]  # 使数据闭合
stats_external_modal += stats_external_modal[:1]  # 使数据闭合
stats_baseline_modal += stats_baseline_modal[:1]
angles += angles[:1]  # 使角度闭合

# %%
# 创建雷达图
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

# 绘制两条模型线
ax.plot(angles, stats_external_modal, marker='^', markersize=4,
        color='#E0817D', linewidth=1, linestyle='solid', label='External test set')
ax.plot(angles, stats_baseline_modal, marker='s', markersize=4,
        color='#BABABA', linewidth=1, linestyle='solid', label='Baseline')

# 填充模型区域
ax.fill(angles, stats_external_modal, color='#F7F0E8', alpha=0.7)
ax.fill(angles, stats_baseline_modal, color="#BABABA", alpha=0.2)

# 设置轴标签
ax.set_xticks(angles[:-1])  # 不重复最后一个角度

ax.set_xticklabels(labels, fontproperties=prop)

# 获取标签对象列表
xticklabels = ax.get_xticklabels()
# 修改第一个标签的位置 (索引为 0)
xticklabels[0].set_position((0, 0.06)) # 调整 tangential_position

# 手动调整第二个和第五个标签的位置
for i, label in enumerate(ax.get_xticklabels()):
    if i == 0:  
        label.set_position((label.get_position()[0] + 0.1, label.get_position()[1] - 0.18))  # 右移、下移
    elif i == 3: 
        label.set_position((label.get_position()[0] + 0.1, label.get_position()[1] - 0.1))  # 右移、下移

ax.set_yticks(np.linspace(0.5, 0.9, num=5))  # 设置y轴为5个刻度（包括最小值和最大值）
ax.set_ylim(0.5, 0.9)  # 设置雷达图的范围为 [0.4, 1]

# 关键：设置多边形样式（关闭圆形网格）
ax.yaxis.grid(False)  # 关闭圆形网格线
ax.xaxis.grid(True, linestyle='--',  linewidth=0.5, color='gray')  # 设置多边形网格线

# 设置 y 轴范围和刻度（去掉 1.0）
y_ticks = np.linspace(0.5, 0.9, num=5)[:-1]  # 刻度值为 [0.4, 0.6, 0.8]，去掉 1.0
ax.set_yticks(y_ticks,)
# 修改 y 轴刻度标签的字体大小
ax.set_yticklabels(y_ticks, fontproperties=prop, fontsize=8)  # 设置字体大小为 12

# 手动添加刻度线连接（多边形样式，仍然绘制到 1.0）
for value in np.linspace(0.5, 0.9, num=5):  # 绘制网格线到 1.0
    grid_line = [value] * (len(labels) + 1)  # 每个轴上的刻度值
    if value == 0.9:
        ax.plot(angles, grid_line, color='gray', linewidth=0.75, linestyle='-')
    else:
        ax.plot(angles, grid_line, color='gray', linewidth=0.5, linestyle='--')
    
# 设置刻度标签（只显示到 0.8）
ax.set_yticklabels([f'{val:.1f}' for val in y_ticks], color='gray')  # 显示刻度标签

# 关闭圆形网格线
ax.yaxis.grid(False)
ax.spines['polar'].set_visible(False)

# 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(0.3, 1.1), prop=prop, frameon=False)

plt.tight_layout()
plt.savefig('radar_modal compare_external.png', dpi=600)  # dpi 参数控制图片分辨率
#plt.savefig('modal compare-binary.svg', )  # dpi 参数控制图片分辨率

# %%
