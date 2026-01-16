# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

# %%
df_non_fill = pd.read_csv("../data/all_curated_nonfill.csv",keep_default_na=False, na_values=[''])
df_non_fill

# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Primary size (nm)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color="#FBA83B", edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
# plt.xticks(fontsize=30)  
plt.xticks([0,100,200,], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])

plt.savefig('data_visualize_plot/data_visualize_primary_size_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)




# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['DLS size in water (nm)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#FBA83B', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
# plt.xticks(fontsize=30)  
plt.xticks([0,100,200,300], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])

plt.savefig('data_visualize_plot/data_visualize_DLS_water_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)




# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Zeta potential in water (mV)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#FBA83B', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
plt.xticks(fontsize=30)  
plt.xticks([-40,-20,0,20], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Zeta_water_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)





# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['PdI in water']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#FBA83B', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([0.2,0.4,0.6,0.8], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_PdI_water_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)





# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['DLS size in dispersion medium (nm)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#FBA83B', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([0,100,200,300], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_DLS_medium_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)





# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Zeta potential in dispersion medium (mV)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#FBA83B', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([-50,-25,0,25], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Zeta_medium_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)



# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['PdI in dispersion medium']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#FBA83B', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([0.1, 0.2, 0.3, 0.4,], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_PdI_medium_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)







# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['NM concentration']

# 将 'mg/L' 替换为空字符串（返回一个新的 Series）
value_series = value_series.str.replace('mg/L', '', case=False, regex=False)

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#FBA83B', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([0,2000,4000,], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Conc_mg_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)





# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Protein source concentration']

# 将 'mg/L' 替换为空字符串（返回一个新的 Series）
value_series = value_series.str.replace('%', '', case=False, regex=False)

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color="#E7958C", edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([0,25,50,100], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Pro_Conc_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)







# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Incubation time (h)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color="#70D3C2", edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([0.0,6,12,], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_incub_time_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)




# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Centrifugation speed']

# 将 'mg/L' 替换为空字符串（返回一个新的 Series）
value_series = value_series.str.replace('g', '', case=False, regex=False)

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color="#7D76DF", edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
# 定义新的刻度标签
ticks = np.array([0, 40000, 80000, 120000, 160000])
labels = [f'{t/1e3:.0f}' for t in ticks]
plt.xticks(ticks, labels, fontproperties=prop, fontsize=30, color="#555555")
ax.text(0.9, 0.3, r'$\times 10^3$', transform=ax.transAxes,fontproperties=prop, 
        fontsize=24, color="#555555", verticalalignment='center')
ax.text(0.9, 0.9, r'unit: g', transform=ax.transAxes, fontproperties=prop, 
        fontsize=28, color="#555555", verticalalignment='center')

plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Cent_g_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)








# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Centrifugation speed']

# 将 'mg/L' 替换为空字符串（返回一个新的 Series）
value_series = value_series.str.replace('rpm', '', case=False, regex=False)

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#7D76DF', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
# 定义新的刻度标签
plt.xticks([12500,15000,17500,20000],  fontproperties=prop, fontsize=30, color="#555555") 
ax.text(0.8, 0.85, r'unit: rpm', transform=ax.transAxes, fontproperties=prop, 
        fontsize=30, color="#555555", verticalalignment='center')

plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Cent_rpm_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)






# %%
# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Centrifugation time (min)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#7D76DF', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([20,40,60], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Cent_time_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)







# %%
# %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Centrifugation temperature (℃)']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#7D76DF', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([0,12,24,36], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Cent_temp_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)







# %% %%
# 获取原始数据（pandas Series）
value_series = df_non_fill['Centrifugation repetitions']

# 转换为 numeric，无法转换的变为 NaN（仍然是 Series）
numeric_value = pd.to_numeric(value_series, errors='coerce')

# 去掉 NaN（仍然保持为 Series）
numeric_value_clean = numeric_value.dropna()

# 如果你需要 numpy 数组：
numeric_value_array = numeric_value_clean.values
# 计算 3% 和 97% 的分位数
lower_bound = np.percentile(numeric_value_array, 3)
upper_bound = np.percentile(numeric_value_array, 97)

print(f"下限 (3%): {lower_bound}")
print(f"上限 (97%): {upper_bound}")

print(f"最小: {numeric_value_array.min()}")
print(f"最大: {numeric_value_array.max()}")

# 过滤掉小于下限或大于上限的值
filtered_array = numeric_value_array[(numeric_value_array >= lower_bound) & (numeric_value_array <= upper_bound)]
filtered_array

# %%
# 创建绘图区域
plt.figure(figsize=(8, 1))  

# 绘制直方图（可调整柱子颜色 color 参数）
plt.hist(filtered_array, bins=100, color='#7D76DF', edgecolor='none')  # edgecolor='none' 去掉黑色边框

# 去除背景和边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.set_facecolor("white")  # 设置坐标轴区域背景为白色
plt.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
plt.grid(False)  # 关闭网格线
#plt.xticks(fontsize=30)  
plt.xticks([1,3,5,], fontproperties=prop, fontsize=30, color="#555555") 
plt.yticks([])
# 设置标题和坐标轴标签（可选）
#plt.title('Distribution of Primary Size', fontsize=14)
#plt.xlabel('Size', fontsize=12)
#plt.ylabel('Frequency', fontsize=12)
plt.rcParams["axes.unicode_minus"] = False

# 保存图像（透明背景关闭，背景为纯白）
plt.savefig('data_visualize_plot/data_visualize_Cent_reps_hist.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.1, facecolor='none',
            transparent=True)



# %%
