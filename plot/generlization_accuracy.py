# %%
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib import cm, colors
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

import itertools
import networkx as nx

font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

# %%
# 设置根目录为当前目录的上一级目录
root_dir = Path('../') 

# 用于存储找到的文件夹名称的列表
folder_names = []

# 1. 使用 glob('*') 查找所有以 'train_' 开头的条目
# glob() 返回的是 Path 对象
for item in root_dir.glob('train_*'):
    # 2. 检查该条目是否是一个目录 (文件夹)
    if item.is_dir():
        # 3. 提取文件夹的名称（basename）并添加到列表中
        # item.name 属性返回路径的最后一部分（即文件夹名）
        folder_names.append(item.name)

print(f"✅ 在目录 '{root_dir.resolve()}' 下找到以下 'train_' 开头的文件夹:")
# 对列表进行原地排序
folder_names.sort()
print(folder_names)

# %%
# 用于存储所有读取到的DataFrame的列表
all_dataframes = []

# 遍历每个文件夹
for folder in folder_names:
    # 构建搜索路径模式: 
    # folder/output/以 eval_summary 开头的任意 CSV 文件
    # 使用 f-string 构建模式
    search_pattern = f"{folder}/output/eval_summary*.csv" 
    
    # 使用 glob() 查找所有匹配的文件路径
    for file_path in root_dir.glob(search_pattern):
        print(f"Reading file: {file_path}")
        
        try:
            # 使用 pandas 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 可选：添加一列来标识数据来源于哪个文件/文件夹
            df['source_file'] = str(folder)
            
            file_name = file_path.name
            # 提取 eval_summary_ 后面的部分（不含 .csv）
            match = re.search(r"eval_summary_(.+)\.csv", file_name)
            if match:
                df["internal or external"] = match.group(1)
            else:
                df["internal or external"] = "unknown"
            
            # 先把后面的数字部分取出来，比如 "stage_1245" -> "1245"
            digits = df['framework'].str.extract(r'stage_(\d+)', expand=False)

            # 把 stage 变成字符串，方便查找
            stage_str = df['stage'].astype(str)

            # 对每一行根据 stage 找到对应字符的位置并切片
            def get_label(digits_str, s):
                pos = digits_str.find(s)       # 找到当前 stage 对应字符的位置
                if pos == -1:
                    return None               # 找不到就返回 None / NaN，看你需求
                return digits_str[:pos+1]      # 包含当前字符以及之前的字符

            df['Label'] = [
                get_label(d, s) for d, s in zip(digits, stage_str)
            ]

            
            char_replacement_map = {
                '1': 'a',
                '2': 'b',
                '3': 'c',
                '4': 'd',
                '5': 'e',
            }

            # 使用 str.replace() 结合正则表达式进行批量替换
            # regex=True 确保替换适用于所有匹配项，而不是整个单元格
            # 注意：在 Series.str.replace() 中，当使用字典进行多重替换时，默认就是逐个替换。
            df['plot label'] = df['Label'].str.replace(
                '|'.join(re.escape(k) for k in char_replacement_map.keys()), # 创建正则表达式 '0|1|2|...'
                lambda match: char_replacement_map[match.group(0)],          # 查找匹配到的数字，并用字典中的值替换
                regex=True
            )
        
            df = df.drop_duplicates(subset=['Label'], keep='first')
            
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# 将所有 DataFrame 合并成一个
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print("\n✅ All data combined into a single DataFrame.")
    print(combined_df.head()) # 打印合并后的前几行进行检查
else:
    print("\n❌ No matching CSV files found.")
    
# %%
combined_df

# %%
combined_df.groupby("source_file").size()


# %%
plot_item = ['train_basic_10']
plot_item

# %%
df_sub = combined_df[
    (combined_df['source_file'].isin(plot_item)) &
    (combined_df['internal or external'] == 'internal')
]
df_sub

# %%
baseline_auroc = df_sub.loc[
    df_sub['plot label'] == 'a', 'AUROC'
].mean()
baseline_auroc

# %%
stages = ['b', 'c', 'd', 'e']

mean_df = pd.DataFrame(
    np.nan, index=stages, columns=stages
)
std_df = mean_df.copy()
mean_df

# %%
for i in stages:
    for j in stages:
        if i == j:
            # 包含 i
            mask_in = df_sub['plot label'].str.contains(i, na=False)
        else:
            # 同时包含 i 和 j
            mask_in = (
                df_sub['plot label'].str.contains(i, na=False) &
                df_sub['plot label'].str.contains(j, na=False)
            )

        # 不包含（即 mask_in 的反面）
        mask_out = ~mask_in

        auroc_in = df_sub.loc[mask_in, 'AUROC']
        auroc_out = df_sub.loc[mask_out, 'AUROC']

        # 两边都要有数据才有意义
        if len(auroc_in) > 0 and len(auroc_out) > 0:
            print(f"{i} and {j}: in={len(auroc_in)}, out={len(auroc_out)}")

            diff = auroc_in.mean() - auroc_out.mean()
            mean_df.loc[i, j] = diff

            # （可选）给一个“差值的标准误/标准差”估计
            # 这里用 pooled 的方式：Var(mean_in - mean_out) = Var_in/n_in + Var_out/n_out
            std = (auroc_in.var(ddof=1) / len(auroc_in) + auroc_out.var(ddof=1) / len(auroc_out)) ** 0.5
            std_df.loc[i, j] = std
        else:
            mean_df.loc[i, j] = float("nan")
            std_df.loc[i, j] = float("nan")

mean_df


# %%
std_df

# %%
df = mean_df.copy()

# ====== 参数 ======
nodes = list(df.index)
pos = {n: (i, 0) for i, n in enumerate(nodes)}  # 一排放

# 线宽范围（你可以调）
lw_min, lw_max = 2.0, 8.0

# 选择 colormap（可选：viridis / plasma / inferno / magma / coolwarm）
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# ====== 准备颜色映射（按 df 全部值归一化）======
vals = df.values.flatten()
vmin, vmax = vals.min(), vals.max()
vmin = -0.002
norm = colors.Normalize(vmin=vmin, vmax=vmax)


cmap = sns.diverging_palette(220, 10, as_cmap=True)

norm = TwoSlopeNorm(
    vmin=vmin,
    vcenter=0.0,
    vmax=vmax
)

vals = np.abs(df.values.flatten())
vmin_abs, vmax_abs = vals.min(), vals.max()
def get_lw(v):
    """把 |v| 映射到 [lw_min, lw_max]"""
    av = abs(float(v))
    if vmax_abs == vmin_abs:
        return (lw_min + lw_max) / 2
    return lw_min + (lw_max - lw_min) * (av - vmin_abs) / (vmax_abs - vmin_abs)

def get_color(v):
    """把 v 映射到 colormap 颜色"""
    return cmap(norm(v))

def draw_self_loop_circle(ax, x_bottom, y_bottom, lw, color,
                                      r=0.35, alpha=0.95, z=1,
                                      height_scale=6.6, width_scale=2.0,
                                      angle=0, theta1=0, theta2=360):
    """
    传入的 (x_bottom, y_bottom) 是“环的最低点”
    r: 控制大小的基准
    height_scale: 椭圆高度 = height_scale * r
    width_scale:  椭圆宽度 = width_scale  * r
    """
    W = width_scale * r / 2
    H = height_scale * r / 5

    # 把最低点换算成圆心/椭圆心
    cx = x_bottom
    cy = y_bottom + H / 2.0

    arc = Arc(
        (cx, cy),
        width=W, height=H,
        angle=angle,
        theta1=theta1, theta2=theta2,
        lw=lw,
        color=color,
        alpha=alpha,
        zorder=z
    )
    arc.set_clip_on(False)
    ax.add_patch(arc)
    
    
# ====== 开始画图 ======
# b/c/d/e 的解释
stage_desc = {
    "b": "Cross-biofluid\naugmentation",
    "c": "Cross-species\naugmentation",
    "d": "Far-domain\naugmentation",
    "e": "In-domain\nrefinement",
}
y_offset = 0.65  # 文字往上抬多少（可调）


# ====== 开始画图 ======
fig, ax = plt.subplots(figsize=(8.5, 3))

# 1) 画节点：白底黑边黑字
for n, (x, y) in pos.items():
    ax.scatter(
        x, y,
        s=1200,
        facecolors="white",   # 白底
        edgecolors="gray",   # 黑边
        linewidths=2.2,       # 边框粗细（可调）
        zorder=3
    )
    ax.text(
        x, y, n,
        ha="center", va="center",
        fontsize=14,fontproperties=prop,
        color="black",        # 黑字
        zorder=4
    )
    
    # 上方说明（只给 b/c/d/e 加）
    if n in stage_desc:
        ax.text(
            x, y + y_offset, stage_desc[n],
            ha="center", va="bottom",
            fontsize=12, fontproperties=prop,
            color="black",
            zorder=4
        )

ax.text(
    0.07, 0.5, "Internal test set",   # x<0 放到左侧外边
    transform=ax.transAxes,            # 用 axes 坐标（稳定）
    rotation=90,                       # 竖着
    ha="center", va="center",
    fontsize=12, fontproperties=prop,
    color="black",
    zorder=12
) 
  
# 2) 画边（无向：只画上三角 + 对角）
for i, ni in enumerate(nodes):
    for j, nj in enumerate(nodes):
        if j < i:
            continue  # 避免重复（只画上三角）

        v = float(df.loc[ni, nj])
        lw = get_lw(v)
        col = get_color(v)

        xi, yi = pos[ni]
        xj, yj = pos[nj]

        # --- 自环 ---
        if ni == nj:
            draw_self_loop_circle(ax, xi, yi, lw=lw, color=col, r=0.40)
            continue

        # --- 普通边 ---
        dist = abs(xj - xi)
        rad = 0.12 * dist  # 距离越远，弧度越大

        edge = FancyArrowPatch(
            (xi, yi), (xj, yj),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",      # 无箭头（无向）
            lw=lw,
            color=col,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(edge)

# 3) 添加颜色条（说明颜色对应数值大小）
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax, fraction=0.03, shrink=0.6, pad=-0.08)
cbar.set_label("Main effect (ΔAUROC)",fontproperties=prop, fontsize=12, labelpad=18)
cbar.ax.yaxis.label.set_rotation(270)
# tick labels 字体
for tick in cbar.ax.get_yticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)   # 可单独控制刻度字号
    
# 4) 美化
ax.set_xlim(-0.7, len(nodes) - 0.3)
ax.set_ylim(-0.9, 1.2)
ax.axis("off")
plt.tight_layout()
plt.savefig('stage_main_effect_internal.png', dpi=600,  bbox_inches='tight') 

plt.show()





# %%
plot_item = ['train_date_10']
plot_item

# %%
df_sub = combined_df[
    (combined_df['source_file'].isin(plot_item)) &
    (combined_df['internal or external'] == 'external')
]
df_sub

# %%
baseline_auroc = df_sub.loc[
    df_sub['plot label'] == 'a', 'AUROC'
].mean()
baseline_auroc

# %%
stages = ['b', 'c', 'd', 'e']

mean_df = pd.DataFrame(
    np.nan, index=stages, columns=stages
)
std_df = mean_df.copy()
mean_df

# %%
for i in stages:
    for j in stages:
        if i == j:
            # 包含 i
            mask_in = df_sub['plot label'].str.contains(i, na=False)
        else:
            # 同时包含 i 和 j
            mask_in = (
                df_sub['plot label'].str.contains(i, na=False) &
                df_sub['plot label'].str.contains(j, na=False)
            )

        # 不包含（即 mask_in 的反面）
        mask_out = ~mask_in

        auroc_in = df_sub.loc[mask_in, 'AUROC']
        auroc_out = df_sub.loc[mask_out, 'AUROC']

        # 两边都要有数据才有意义
        if len(auroc_in) > 0 and len(auroc_out) > 0:
            print(f"{i} and {j}: in={len(auroc_in)}, out={len(auroc_out)}")

            diff = auroc_in.mean() - auroc_out.mean()
            mean_df.loc[i, j] = diff

            # （可选）给一个“差值的标准误/标准差”估计
            # 这里用 pooled 的方式：Var(mean_in - mean_out) = Var_in/n_in + Var_out/n_out
            std = (auroc_in.var(ddof=1) / len(auroc_in) + auroc_out.var(ddof=1) / len(auroc_out)) ** 0.5
            std_df.loc[i, j] = std
        else:
            mean_df.loc[i, j] = float("nan")
            std_df.loc[i, j] = float("nan")

mean_df


# %%
std_df

# %%
df = mean_df.copy()

# ====== 参数 ======
nodes = list(df.index)
pos = {n: (i, 0) for i, n in enumerate(nodes)}  # 一排放

# 线宽范围（你可以调）
lw_min, lw_max = 2.0, 8.0

# 选择 colormap（可选：viridis / plasma / inferno / magma / coolwarm）
cmap = sns.diverging_palette(220, 10, as_cmap=True)

def truncate_cmap(cmap, minval=0.5, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        'trunc_cmap',
        cmap(np.linspace(minval, maxval, n))
    )

cmap = truncate_cmap(cmap, 0.5, 1.0)


# ====== 准备颜色映射（按 df 全部值归一化）======
vals = df.values.flatten()
vmin, vmax = vals.min(), vals.max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)


vals = np.abs(df.values.flatten())
vmin_abs, vmax_abs = vals.min(), vals.max()
def get_lw(v):
    """把 |v| 映射到 [lw_min, lw_max]"""
    av = abs(float(v))
    if vmax_abs == vmin_abs:
        return (lw_min + lw_max) / 2
    return lw_min + (lw_max - lw_min) * (av - vmin_abs) / (vmax_abs - vmin_abs)

def get_color(v):
    """把 v 映射到 colormap 颜色"""
    return cmap(norm(v))

def draw_self_loop_circle(ax, x_bottom, y_bottom, lw, color,
                                      r=0.35, alpha=0.95, z=1,
                                      height_scale=6.6, width_scale=2.0,
                                      angle=0, theta1=0, theta2=360):
    """
    传入的 (x_bottom, y_bottom) 是“环的最低点”
    r: 控制大小的基准
    height_scale: 椭圆高度 = height_scale * r
    width_scale:  椭圆宽度 = width_scale  * r
    """
    W = width_scale * r / 2
    H = height_scale * r / 5

    # 把最低点换算成圆心/椭圆心
    cx = x_bottom
    cy = y_bottom + H / 2.0

    arc = Arc(
        (cx, cy),
        width=W, height=H,
        angle=angle,
        theta1=theta1, theta2=theta2,
        lw=lw,
        color=color,
        alpha=alpha,
        zorder=z
    )
    arc.set_clip_on(False)
    ax.add_patch(arc)
    
# ====== 开始画图 ======
fig, ax = plt.subplots(figsize=(8.5, 3))

# 1) 画节点：白底黑边黑字
for n, (x, y) in pos.items():
    ax.scatter(
        x, y,
        s=1200,
        facecolors="white",   # 白底
        edgecolors="gray",   # 黑边
        linewidths=2.2,       # 边框粗细（可调）
        zorder=3
    )
    ax.text(
        x, y, n,
        ha="center", va="center",
        fontsize=14,fontproperties=prop,
        color="black",        # 黑字
        zorder=4
    )

ax.text(
    0.07, 0.5, "External test set",   # x<0 放到左侧外边
    transform=ax.transAxes,            # 用 axes 坐标（稳定）
    rotation=90,                       # 竖着
    ha="center", va="center",
    fontsize=12, fontproperties=prop,
    color="black",
    zorder=12
) 
    
# 2) 画边（无向：只画上三角 + 对角）
for i, ni in enumerate(nodes):
    for j, nj in enumerate(nodes):
        if j < i:
            continue  # 避免重复（只画上三角）

        v = float(df.loc[ni, nj])
        lw = get_lw(v)
        col = get_color(v)

        xi, yi = pos[ni]
        xj, yj = pos[nj]

        # --- 自环 ---
        if ni == nj:
            draw_self_loop_circle(ax, xi, yi, lw=lw, color=col, r=0.40)
            continue

        # --- 普通边 ---
        dist = abs(xj - xi)
        rad = 0.12 * dist  # 距离越远，弧度越大

        edge = FancyArrowPatch(
            (xi, yi), (xj, yj),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",      # 无箭头（无向）
            lw=lw,
            color=col,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(edge)

# 3) 添加颜色条（说明颜色对应数值大小）
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, shrink=0.7, pad=-0.08)
cbar.set_label("Main effect (ΔAUROC)",fontproperties=prop, fontsize=12, labelpad=18)
cbar.ax.yaxis.label.set_rotation(270)
# tick labels 字体
for tick in cbar.ax.get_yticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)   # 可单独控制刻度字号
    
# 4) 美化
ax.set_xlim(-0.7, len(nodes) - 0.3)
ax.set_ylim(-0.9, 1.2)
ax.axis("off")
plt.tight_layout()
plt.savefig('stage_main_effect_external.png', dpi=600,  bbox_inches='tight',transparent=True) 

plt.show()

# %%
