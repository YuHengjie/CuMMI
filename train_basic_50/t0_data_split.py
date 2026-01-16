# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re

# %%
df = pd.read_csv("../data/all_curated_nonfill.csv",keep_default_na=False, na_values=[''])
df

# %%
df["RPA"].hist(bins=200)

# %%
rpa_values = df['RPA'].values
threshold = 1e-6
count_below_threshold = (rpa_values < threshold).sum()  # 统计小于 threshold 的样本数量
total_count = len(rpa_values)  # 总样本数量
percentage_below_threshold = count_below_threshold / total_count * 100  # 计算占比

print(f"Number of samples with RPA < threshold: {count_below_threshold}")
print(f"Total number of samples: {total_count}")
print(f"Percentage of samples with RPA < threshold: {percentage_below_threshold:.2f}%")

# %%
df['Protein corona composition'] = df['RPA'].apply(lambda x: 0 if x < threshold else 1)
sum(df['Protein corona composition'])

# %%
df['Incubation protein source'].unique()

# %%
df['Protein source organism'].unique()

# %%
mask = ~df['Incubation protein source'].str.contains(
    'serum|plasma|blood', case=False, na=False
)
df_not_blood = df[mask]
df_not_blood['Incubation protein source'].unique()
print(f"筛选到 {len(df_not_blood)} 条记录。")

# %%
mask_source = df['Incubation protein source'].str.contains(
    'serum|plasma|blood', case=False, na=False
)
mask_not_human = ~df['Protein source organism'].str.contains(
    'human', case=False, na=False
)

df_blood_nonhuman = df[mask_source & mask_not_human]

print(df_blood_nonhuman['Protein source organism'].unique())
print(f"筛选到 {len(df_blood_nonhuman)} 条记录。")


# %%
mask_serum = df['Incubation protein source'].str.contains(
    'serum', case=False, na=False
)
mask_human = df['Protein source organism'].str.contains(
    'human', case=False, na=False
)

df_serum_human = df[mask_serum & mask_human]

print(df_serum_human['Protein source organism'].unique())
print(f"筛选到 {len(df_serum_human)} 条记录。")

# %%
mask_plasma = df['Incubation protein source'].str.contains(
    'plasma', case=False, na=False
)
mask_human = df['Protein source organism'].str.contains(
    'human', case=False, na=False
)

df_plasma_human = df[mask_plasma & mask_human]

print(df_plasma_human['Protein source organism'].unique())
print(f"筛选到 {len(df_plasma_human)} 条记录。")

# %%
prob_label = pd.read_csv("../data/problematic_labels.csv",)
prob_label

# %%
# 从 prob_label 取出 Label 列的唯一值
problematic_labels = prob_label['Label'].dropna().unique()

# %%
df_plasma_human_low = df_plasma_human[df_plasma_human['Label'].isin(problematic_labels)]
print(f"筛选到 {len(df_plasma_human_low)} 条记录。")

# %%
df_plasma_human_high = df_plasma_human[~df_plasma_human['Label'].isin(problematic_labels)]
print(f"筛选到 {len(df_plasma_human_high)} 条记录。")

# %%
# 按反斜杠分割，然后找出每行中包含 '10.' 的部分
doi_series = (
    df_plasma_human_high['Label']
    .str.split('\\')                                   # 分割路径
    .apply(lambda parts: next((p for p in parts if isinstance(p, str) and p.startswith('10.')), None))
)

# 显示前几行和唯一 DOI 数量
print(doi_series)
print("Unique DOI count:", doi_series.nunique())

# %%
# 使用 Series.str.replace(pat, repl, n=1) 确保只替换第一次出现的 '_'
processed_doi_series = doi_series.astype(str).str.replace('_', '/', n=1)
# 获取 unique DOI，并移除 None/NaN 值
unique_dois = processed_doi_series.dropna().unique()
print(len(unique_dois))
# 定义输出文件名
output_filename = 'unique_processed_dois.txt'

# 将唯一 DOI 数组写入文件，每行一个
try:
    with open(output_filename, 'w') as f:
        # 使用 join 和换行符写入
        f.write('\n'.join(unique_dois))
    
    print(f"\n✅ 成功将 {len(unique_dois)} 个独特 DOI 保存到文件: {output_filename}")
    
except Exception as e:
    print(f"\n❌ 写入文件时发生错误: {e}")
    

# %%
# 打乱 df_plasma_human_high 数据框
df_plasma_human_high = df_plasma_human_high.sample(frac=1, random_state=42)

# 从 df_plasma_human_high 中随机抽取 15% 作为测试集
df_plasma_human_high_test = df_plasma_human_high.sample(frac=0.15, random_state=42)

# 从剩余的 85% 中随机抽取 15% 作为验证集
df_plasma_human_high_remaining = df_plasma_human_high.drop(df_plasma_human_high_test.index)
df_plasma_human_high_val = df_plasma_human_high_remaining.sample(frac=0.1764706, random_state=42)  # 15% of the original dataset

# 剩下的 70% 作为训练集
df_plasma_human_high_train = df_plasma_human_high_remaining.drop(df_plasma_human_high_val.index)

# %%
# 打乱 df_not_blood 数据框
df_not_blood = df_not_blood.sample(frac=1, random_state=42)

# %%
# 打乱 df_blood_nonhuman 数据框
df_blood_nonhuman = df_blood_nonhuman.sample(frac=1, random_state=42)

# %% 合并打乱df_serum_human_plasma_low
df_serum_human_plasma_low = pd.concat([df_serum_human, df_plasma_human_low], )
df_serum_human_plasma_low = df_serum_human_plasma_low.sample(frac=1, random_state=42)
df_serum_human_plasma_low


# %%
#df_plasma_human_high_final = df_plasma_human_high_train[df_plasma_human_high_train['Overall data quality'] > 0.7]
df_plasma_human_high_final = df_plasma_human_high_train.sample(
    frac=0.5, 
    random_state=42 # 建议使用固定的随机种子
)

# %%
# 按反斜杠分割，然后找出每行中包含 '10.' 的部分
doi_series = (
    df_plasma_human_high_final['Label']
    .str.split('\\')                                   # 分割路径
    .apply(lambda parts: next((p for p in parts if isinstance(p, str) and p.startswith('10.')), None))
)

# 显示前几行和唯一 DOI 数量
print(doi_series)
print("Unique DOI count:", doi_series.nunique())

# %%
# 确保 data 文件夹存在
os.makedirs("data", exist_ok=True)

# 将各个 DataFrame 的索引保存为字典
index_dict = {
    "not_blood": df_not_blood.index.tolist(),
    "blood_nonhuman": df_blood_nonhuman.index.tolist(),
    "serum_human_plasma_low": df_serum_human_plasma_low.index.tolist(),
    
    "plasma_human_high": df_plasma_human_high.index.tolist(),
    "plasma_human_high_train": df_plasma_human_high_train.index.tolist(),
    "plasma_human_high_val": df_plasma_human_high_val.index.tolist(),
    "plasma_human_high_test": df_plasma_human_high_test.index.tolist(),

    "plasma_human_high_final": df_plasma_human_high_final.index.tolist()
}

# 保存为 JSON 文件
with open("data/data_split_indices.json", "w", encoding="utf-8") as f:
    json.dump(index_dict, f, ensure_ascii=False, indent=4)

print("✅ 索引已保存到 data/data_split_indices.json")

# %%
for name, idx_list in index_dict.items():
    print(f"{name}: {len(idx_list)}")


# %%
# 从 JSON 文件中读取索引
with open("data/data_split_indices.json", "r", encoding="utf-8") as f:
    index_dict = json.load(f)

# %%
# 根据 index_dict 批量在 df 上取子集
df_not_blood = df.loc[index_dict["not_blood"]]
df_blood_nonhuman = df.loc[index_dict["blood_nonhuman"]]
df_serum_human_plasma_low = df.loc[index_dict["serum_human_plasma_low"]]

df_plasma_human_high = df.loc[index_dict["plasma_human_high"]]
df_plasma_human_high_train = df.loc[index_dict["plasma_human_high_train"]]
df_plasma_human_high_val = df.loc[index_dict["plasma_human_high_val"]]
df_plasma_human_high_test = df.loc[index_dict["plasma_human_high_test"]]

df_plasma_human_high_final = df.loc[index_dict["plasma_human_high_final"]]

# %%
# 打印各数据集大小确认
for name, subset in {
    "not_blood": df_not_blood,
    "blood_nonhuman": df_blood_nonhuman,
    "serum_human_plasma_low": df_serum_human_plasma_low,
    "plasma_human_high": df_plasma_human_high,
    "plasma_human_high_train": df_plasma_human_high_train,
    "plasma_human_high_val": df_plasma_human_high_val,
    "plasma_human_high_test": df_plasma_human_high_test,
    "plasma_human_high_final": df_plasma_human_high_final
}.items():
    print(f"{name}: {len(subset)}")
    
# %%
# 确保目标目录存在
os.makedirs("data", exist_ok=True)

combined_dfs = {}
for key in index_dict.keys():
    df_name = f"df_{key}"

    if df_name in globals():
        df_to_save = globals()[df_name]
        combined_dfs[key] = df_to_save # 
        save_path = f"data/basic_{key}.csv"
        df_to_save.to_csv(save_path, index=False)
        print(f"{key}: 保存 df_{key}，共 {len(df_to_save)} 条记录，已保存到 {save_path}")


# %% 增加远域数据的100%
# 复制 not_blood 数据
df_nb = combined_dfs["not_blood"].copy()

# 从 plasma_human_high 中随机抽取等数量的数据
n_nb = len(df_nb)
# 注意：使用 combined_dfs["plasma_human_high"] 作为抽样池，确保数据一致性
df_phh_nb = combined_dfs["plasma_human_high"].sample(
    n=n_nb, replace=False, random_state=42
).copy()

# 合并并打乱
df_nb_addhigh = pd.concat([df_nb, df_phh_nb], axis=0, ignore_index=True)
df_nb_addhigh = df_nb_addhigh.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
save_path_nb = "data/basic_not_blood_addhigh100.csv"
df_nb_addhigh.to_csv(save_path_nb, index=False)

print(f"✅ 已保存 {save_path_nb}，"
      f"not_blood 原始 {len(df_nb)} 条，"
      f"plasma_human_high 抽样 {len(df_phh_nb)} 条，"
      f"合并后共 {len(df_nb_addhigh)} 条。")

train_nb, val_nb = train_test_split(
    df_nb_addhigh, test_size=0.15, random_state=42, shuffle=True
)

train_nb.to_csv("data/basic_not_blood_addhigh100_train.csv", index=False)
val_nb.to_csv("data/basic_not_blood_addhigh100_val.csv", index=False)

print(f"✅ not_blood_addhigh 划分完成：train={len(train_nb)}, val={len(val_nb)}")

# %% 增加远域数据的100%
# 复制 blood_nonhuman 数据
df_bn = combined_dfs["blood_nonhuman"].copy()

# 从 plasma_human_high 中随机抽取等数量的数据
n = len(df_bn)
df_phh = combined_dfs["plasma_human_high"].sample(
    n=n, replace=False, random_state=42
).copy()

# 合并并打乱
df_bn_addhigh = pd.concat([df_bn, df_phh], axis=0, ignore_index=True)
df_bn_addhigh = df_bn_addhigh.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
save_path = "data/basic_blood_nonhuman_addhigh100.csv"
df_bn_addhigh.to_csv(save_path, index=False)

print(f"✅ 已保存 data/basic_blood_nonhuman_addhigh100.csv，"
      f"blood_nonhuman 原始 {len(df_bn)} 条，"
      f"plasma_human_high 抽样 {len(df_phh)} 条，"
      f"合并后共 {len(df_bn_addhigh)} 条。")

# === 对 blood_nonhuman_addhigh 进行 85/15 划分 ===
train_bn, val_bn = train_test_split(
    df_bn_addhigh, test_size=0.15, random_state=42, shuffle=True
)

train_bn.to_csv("data/basic_blood_nonhuman_addhigh100_train.csv", index=False)
val_bn.to_csv("data/basic_blood_nonhuman_addhigh100_val.csv", index=False)

print(f"✅ blood_nonhuman_addhigh 划分完成：train={len(train_bn)}, val={len(val_bn)}")


# %% 增加近域数据的100%
# 复制 serum_human_plasma_low 数据
df_shpl = combined_dfs["serum_human_plasma_low"].copy()
df_phh = combined_dfs["plasma_human_high"].copy()
# 从 plasma_human_high 中随机抽取 serum 的 100%
m = len(df_shpl)
k = m
high_count = len(df_phh)

# 1. 判断是否大于总样本量
if k > high_count:
    # 目标抽取量大于总样本量：使用有放回抽样 (replace=True)
    print(f"Warning: Target sample size ({k}) exceeds total population ({high_count}). Switching to replacement sampling.")
    df_phh_sample = df_phh.sample(
        n=k, 
        replace=True,  # 有放回抽样 (过采样)
        random_state=42
    ).copy()
else:
    # 目标抽取量小于或等于总样本量：使用无放回抽样 (replace=False)
    print(f"Target sample size ({k}) is safe. Using non-replacement sampling.")
    df_phh_sample = df_phh.sample(
        n=k, 
        replace=False, # 无放回抽样 (正常采样)
        random_state=42
    ).copy()

# 合并并打乱
df_shpl_addhigh100 = pd.concat([df_shpl, df_phh_sample], axis=0, ignore_index=True)
df_shpl_addhigh100 = df_shpl_addhigh100.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
save_path = "data/basic_serum_human_plasma_low_addhigh100.csv"
df_shpl_addhigh100.to_csv(save_path, index=False)

print(
    f"✅ 已保存 {save_path}，"
    f"serum_human_plasma_low 原始 {len(df_shpl)} 条，"
    f"plasma_human_high 抽样 {len(df_phh_sample)} 条，"
    f"合并后共 {len(df_shpl_addhigh100)} 条。"
)

# === 对 serum_human_plasma_low_addhigh100 进行 85/15 划分 ===
train_shpl, val_shpl = train_test_split(
    df_shpl_addhigh100, test_size=0.15, random_state=42, shuffle=True
)

train_shpl.to_csv("data/basic_serum_human_plasma_low_addhigh100_train.csv", index=False)
val_shpl.to_csv("data/basic_serum_human_plasma_low_addhigh100_val.csv", index=False)

print(f"✅ serum_human_plasma_low_addhigh100 划分完成：train={len(train_shpl)}, val={len(val_shpl)}")

# %%
