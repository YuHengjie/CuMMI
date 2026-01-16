# %%
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

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
plot_item = ['train_basic_10','train_basic_30','train_basic_50','train_basic_75','train_basic_100']
plot_item

# %%
df_sub = combined_df[
    (combined_df['source_file'].isin(plot_item)) &
    (combined_df['internal or external'] == 'internal')
]
df_sub

# %%
legend_name_map = {
    'train_basic_10':  '10%',
    'train_basic_30':  '30%',
    'train_basic_50':  '50%',
    'train_basic_75':  '75%',
    'train_basic_100': '100%',
}

df_sub = df_sub.copy()
df_sub['Data proportion'] = df_sub['source_file'].map(legend_name_map).fillna(df_sub['source_file'])
df_sub

# %%
df_sub.loc[df_sub["stage"] != 5, "Data proportion"] = "without"
df_sub

# %%
df_sub = df_sub.drop_duplicates(subset=["Label", "Data proportion"])
df_sub

# %%
df_sub = df_sub[df_sub['Data proportion'].isin(['10%', 'without'])]
df_sub

# %%
plot_item = sorted(df_sub['plot label'].unique())
plot_item

# %%
plot_labels = sorted(df_sub['plot label'].unique())

special_labels = {'a', 'ae'}
color_map = {
    lbl: '#808080' if lbl in special_labels else '#F47F72'
    for lbl in plot_labels
}

marker_map = {
    lbl: 's' if lbl in special_labels else 'o'
    for lbl in plot_labels
}

size_map = {
    lbl: 90 if lbl in special_labels else 120
    for lbl in plot_labels
}


# 给每个 label 一个 y 偏移（同侧按F1排序，居中展开）
# 你手动指定：每个唯一 plot label -> (x_text, y_text)
fixed_text_xy = {
    'a': (0.956, 0.864),
    'ae': (0.958, 0.857),
    'ade': (0.968, 0.883),
    'ace': (0.9715, 0.875),
    'abe': (0.966, 0.878),
    'ab': (0.970, 0.872),

    'ac': (0.973, 0.877),
    'abc': (0.971, 0.888),
    'abce': (0.969, 0.887),
    'acde': (0.9745, 0.876),
    'abde': (0.9715, 0.892),
    
    'abcde': (0.976, 0.880),
    'ad': (0.976, 0.885),
    'abcd': (0.978, 0.887),
    'acd': (0.979, 0.891),

    'abd': (0.9735, 0.895),
}

# 可选：指定对齐方式（不写默认 left）
fixed_ha = {
    'a': 'center',
    'ae': 'center',
    'ade': 'center',
    'ace': 'center',
    'abe': 'center',
    'ab': 'center',
    'ac': 'center',
    'abc': 'center',
    'abce': 'center',
    'abde': 'center',
    'acde': 'center',
    'abcde': 'center',
    'ad': 'center',
    'abcd': 'center',
    'acd': 'center',
    'abd': 'center',
}

# 检查有没有漏填
uniq_labels = set(df_sub['plot label'].unique())
missing = uniq_labels - set(fixed_text_xy.keys())
if missing:
    raise ValueError(f"fixed_text_xy 缺少这些 plot label 的坐标：{sorted(missing)}")

# 组装 label_pos（供后面画线/画字用）
label_pos = {lbl: (*fixed_text_xy[lbl], fixed_ha.get(lbl, 'left')) for lbl in fixed_text_xy}

# =============================
# 2) 绘图：点按组着色；每个点连到对应label的“黑色文本锚点”
# =============================
plt.figure(figsize=(6.5, 4))


for lbl, group in df_sub.groupby('plot label'):
    s = plt.scatter(group['AUROC'], group['F1'],
                    c=color_map[lbl],
                    s=size_map.get(lbl, 120),  
                    marker=marker_map[lbl],
                    edgecolors='white',
                    alpha=0.9,
                    label=lbl,
                    zorder=2)

    # 画连接线：每个点 -> 该 label 的统一文本位置（线颜色=组颜色）
    for _, row in group.iterrows():
        lbl = row['plot label']
        x_text, y_text, _ = label_pos[lbl]
        plt.plot([row['AUROC'], x_text],
                 [row['F1'],    y_text],
                 color=color_map[lbl],
                 lw=1,
                 alpha=1,
                 zorder=1)

# =============================
# 3) 只画一次黑色文本（每个唯一label）
# =============================
for lbl, (x_text, y_text, ha) in label_pos.items():
    plt.text(
        x_text, y_text, lbl,
        ha=ha, va='center',
        fontproperties=prop,
        fontsize=12,
        color="black",                          # ← 动态颜色！
        bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=0.2),
        zorder=3
    )

# 图形修饰
#plt.title("Internal test set performance", fontproperties=prop, fontsize=14)
plt.xlabel("AUROC", fontproperties=prop, fontsize=12)
plt.ylabel("F1", fontproperties=prop, fontsize=12)


# 两类的样式（直接取一个代表label的样式）
lbl_wo = next(iter(special_labels))  # e.g. 'a'
lbl_w  = next(lbl for lbl in plot_labels if lbl not in special_labels)  # 任意一个非special

legend_elements = [
    Line2D(
        [0], [0],
        marker=marker_map[lbl_wo],
        linestyle='None',
        markerfacecolor=color_map[lbl_wo],
        markeredgecolor='white',
        markersize=10,
        label='without'
    ),
    Line2D(
        [0], [0],
        marker=marker_map[lbl_w],
        linestyle='None',
        markerfacecolor=color_map[lbl_w],
        markeredgecolor='white',
        markersize=10,
        label='with'
    ),
]

leg = plt.legend(
    handles=legend_elements,
    prop=prop, fontsize=13,
    title="Auxiliary domain",
    frameon=False
)

# ---- 强制设置 legend 标题字体/字号/居中（通吃各版本）----
title = leg.get_title()
title.set_fontproperties(prop)
title.set_fontsize(12)
title.set_ha("center")

plt.xticks(fontproperties=prop, fontsize=12)
plt.yticks(fontproperties=prop, fontsize=12)

plt.ylim((0.855,0.898))

#plt.grid(True)
plt.tight_layout()
plt.savefig("t1_stage_e_prop_internal_only_10e.png", dpi=600, bbox_inches="tight", transparent=False)
plt.show()


# %%
#plot_item = ['train_date_30']
plot_item = ['train_date_10','train_date_30','train_date_50','train_date_75','train_date_100']
plot_item

# %%
df_sub = combined_df[
    (combined_df['source_file'].isin(plot_item)) &
    (combined_df['internal or external'] == 'external')
]
df_sub

# %%
legend_name_map = {
    'train_date_10':  '10%',
    'train_date_30':  '30%',
    'train_date_50':  '50%',
    'train_date_75':  '75%',
    'train_date_100': '100%',
}

df_sub = df_sub.copy()
df_sub['Data proportion'] = df_sub['source_file'].map(legend_name_map).fillna(df_sub['source_file'])


# %%
df_sub.loc[df_sub["stage"] != 5, "Data proportion"] = "without"
df_sub

# %%
df_sub = df_sub.drop_duplicates(subset=["Label", "Data proportion"])
df_sub

# %%
df_sub = df_sub[df_sub['Data proportion'].isin(['10%', 'without'])]
df_sub

# %%
plot_item = ['10%','30%','50%','75%','100%', 'without']
plot_item

# %%
plot_labels = sorted(df_sub['plot label'].unique())

special_labels = {'a', 'ae'}
color_map = {
    lbl: '#808080' if lbl in special_labels else '#F47F72'
    for lbl in plot_labels
}

marker_map = {
    lbl: 's' if lbl in special_labels else 'o'
    for lbl in plot_labels
}

size_map = {
    lbl: 90 if lbl in special_labels else 120
    for lbl in plot_labels
}

# 给每个 label 一个 y 偏移（同侧按F1排序，居中展开）
# 你手动指定：每个唯一 plot label -> (x_text, y_text)
fixed_text_xy = {
    'a': (0.567, 0.66),
    'ae': (0.59, 0.62),
    'ade': (0.645, 0.72),
    'ace': (0.68, 0.715),
    
    'abe': (0.62, 0.61),
    'ab': (0.67, 0.615),

    'ac': (0.645, 0.705),
    'abc': (0.695, 0.728),
    'abce': (0.71, 0.73),
    
    'acde': (0.65, 0.765),
    'abde': (0.675, 0.68),

    'abcde': (0.72, 0.748),
    'ad': (0.66, 0.73),
    
    'abcd': (0.67, 0.765),
    'acd': (0.615, 0.70),

    'abd': (0.64, 0.665),
}

# 可选：指定对齐方式（不写默认 left）
fixed_ha = {
    'a': 'center',
    'ae': 'center',
    'ade': 'center',
    'ace': 'center',
    'abe': 'center',
    'ab': 'center',
    'ac': 'center',
    'abc': 'center',
    'abce': 'center',
    'abde': 'center',
    'acde': 'center',
    'abcde': 'center',
    'ad': 'center',
    'abcd': 'center',
    'acd': 'center',
    'abd': 'center',
}

# 检查有没有漏填
uniq_labels = set(df_sub['plot label'].unique())
missing = uniq_labels - set(fixed_text_xy.keys())
if missing:
    raise ValueError(f"fixed_text_xy 缺少这些 plot label 的坐标：{sorted(missing)}")

# 组装 label_pos（供后面画线/画字用）
label_pos = {lbl: (*fixed_text_xy[lbl], fixed_ha.get(lbl, 'left')) for lbl in fixed_text_xy}



# =============================
# 2) 绘图：点按组着色；每个点连到对应label的“黑色文本锚点”
# =============================
plt.figure(figsize=(6.5, 4))

for lbl, group in df_sub.groupby('plot label'):
    s = plt.scatter(group['AUROC'], group['F1'],
                    c=color_map[lbl],
                    s=size_map.get(lbl, 120),  
                    marker=marker_map[lbl],
                    edgecolors='white',
                    alpha=0.9,
                    label=lbl,
                    zorder=2)

    # 画连接线：每个点 -> 该 label 的统一文本位置（线颜色=组颜色）
    for _, row in group.iterrows():
        lbl = row['plot label']
        x_text, y_text, _ = label_pos[lbl]
        plt.plot([row['AUROC'], x_text],
                 [row['F1'],    y_text],
                 color=color_map[lbl],
                 lw=1,
                 alpha=0.9,
                 zorder=1)


for lbl, (x_text, y_text, ha) in label_pos.items():
    color = "red" if lbl == 'abcde' else "black"

    plt.text(
        x_text, y_text, lbl,
        ha=ha, va='center',
        fontproperties=prop,
        fontsize=12,
        color=color,                          # ← 动态颜色！
        bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=0.2),
        zorder=3
    )


# 两类的样式（直接取一个代表label的样式）
lbl_wo = next(iter(special_labels))  # e.g. 'a'
lbl_w  = next(lbl for lbl in plot_labels if lbl not in special_labels)  # 任意一个非special

legend_elements = [
    Line2D(
        [0], [0],
        marker=marker_map[lbl_wo],
        linestyle='None',
        markerfacecolor=color_map[lbl_wo],
        markeredgecolor='white',
        markersize=10,
        label='without'
    ),
    Line2D(
        [0], [0],
        marker=marker_map[lbl_w],
        linestyle='None',
        markerfacecolor=color_map[lbl_w],
        markeredgecolor='white',
        markersize=10,
        label='with'
    ),
]

leg = plt.legend(
    handles=legend_elements,
    prop=prop, fontsize=13,
    title="Auxiliary domain",
    frameon=False
)


# 图形修饰
#plt.title("External test set performance", fontproperties=prop, fontsize=14)
plt.xlabel("AUROC", fontproperties=prop, fontsize=12)
plt.ylabel("F1", fontproperties=prop, fontsize=12)



# ---- 强制设置 legend 标题字体/字号/居中（通吃各版本）----
title = leg.get_title()
title.set_fontproperties(prop)
title.set_fontsize(12)
title.set_ha("center")

plt.xticks(fontproperties=prop, fontsize=12)
plt.yticks(fontproperties=prop, fontsize=12)

df_sub['_score'] = df_sub['AUROC'] + df_sub['F1']
row_best = df_sub.loc[df_sub['_score'].idxmax()]

x_best = row_best['AUROC']
y_best = row_best['F1']

# 箭头终点 = 点的位置
x_target = x_best-0.003
y_target = y_best

# 箭头起点 = 往左挪一点，让箭头水平
x_start = x_best - 0.01 # 若尺度不同可调大/调小
y_start = y_best

plt.annotate(
    '', 
    xy=(x_target, y_target),  # 箭头指向的位置
    xytext=(x_start, y_start),  # 箭头起点（左侧）
    arrowprops=dict(
        arrowstyle='->',
        color='red',
        lw=1.25,
    ),
    zorder=10
)

# 可选：在箭头左边写一个说明文字
plt.text(
    x_start - 0.006, y_start, 
    "Best",
    color='red',
    fontsize=12,
    fontproperties=prop,
    ha='center', va='center'
)

#plt.grid(True)
plt.tight_layout()
plt.savefig("t1_stage_e_prop_external_only_10e.png", dpi=600, bbox_inches="tight", transparent=False)
plt.show()

# %%
