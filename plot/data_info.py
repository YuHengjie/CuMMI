# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from matplotlib import font_manager as fm
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

# %%
df_non_fill = pd.read_csv("../data/all_curated_nonfill.csv",keep_default_na=False, na_values=[''])
df_non_fill

# %%
df_non_fill.columns[1:38]

# %%
mask = (
    pd.to_numeric(
        df_non_fill["NM concentration"]
            .astype("string")
            .str.replace("mg/L", "", case=False, regex=False)
            .str.strip(),
        errors="coerce"
    ) < 0.1
)

df_non_fill[mask].loc[:, ["NM concentration"]]

# %%
df_non_fill[mask].loc[:, ["NM concentration"]]["NM concentration"].unique()


# %%
pd.set_option("display.max_rows", None)

# 提前确定的“应该是数值”的特征名列表
numeric_features = [
     "Primary size (nm)", "DLS size in water (nm)", "Zeta potential in water (mV)", 
     "PdI in water", "DLS size in dispersion medium (nm)", "Zeta potential in dispersion medium (mV)",
     "PdI in dispersion medium", "NM concentration", "Protein source concentration",
     "Incubation time (h)",   "Centrifugation speed",
     "Centrifugation time (min)", 
]


feature_analysis = {}

for col in df_non_fill.columns[1:38]:
    s = df_non_fill[col]

    if col in numeric_features:
        # -------- 新增：按列做清洗 + unit --------
        unit = None
        s_for_numeric = s

        if col == "NM concentration":
            # 去掉可能存在的 mg/L（大小写、空格）
            # 用 pandas StringDtype，保留缺失值为 <NA>
            s_for_numeric = (
                s.astype("string")
                 .str.replace("mg/L", "", regex=False)
                 .str.replace("MG/L", "", regex=False)
                 .str.replace("mg/l", "", regex=False)
                 .str.replace("MG/l", "", regex=False)
                 .str.strip()
            )
            unit = "mg/L"

        if col == "Protein source concentration":
            # 去掉可能存在的 %
            s_for_numeric = (
                s.astype("string")
                 .str.replace("%", "", regex=False)
                 .str.strip()
            )
            unit = "%"

        if col == "Centrifugation speed":
            # 去掉可能存在的 %
            s_for_numeric = (
                s.astype("string")
                 .str.replace("g", "", regex=False)
                 .str.strip()
            )
            unit = "g"

    
        # 1) 尝试整体转数值
        numeric = pd.to_numeric(s_for_numeric, errors="coerce")

        # 2) 转不成功的（含原本缺失 NaN）——按原始值逐类统计
        nonnum_mask = numeric.isna()
        nonnum_series = s[nonnum_mask]  # 这里按“原始值”分组统计（保留 mg/L、% 原样也可以对账）

        nonnum_counts = nonnum_series.value_counts(dropna=False)
        nonnum_df = nonnum_counts.reset_index()
        nonnum_df.columns = ["category", "count"]
        nonnum_df["proportion"] = (nonnum_df["count"] / len(s)).round(6)
        nonnum_df["percentage"] = (nonnum_df["proportion"] * 100).round(2)

        # 3) 数值统计（只对成功转数值的部分）
        numeric_valid = numeric.dropna()
        if len(numeric_valid) > 0:
            desc = numeric_valid.describe(percentiles=[0.25, 0.5, 0.75])
            stats_df = pd.DataFrame({
                "statistic": ["min", "max", "mean", "q25", "q50", "q75"],
                "value": [
                    desc["min"],
                    desc["max"],
                    desc["mean"],
                    desc["25%"],
                    desc["50%"],
                    desc["75%"],
                ]
            })
        else:
            stats_df = pd.DataFrame({"statistic": ["no_numeric_values"], "value": ["TRUE"]})

        # 4) 新增：unit 行（放在 stats 下面）
        if unit is not None:
            stats_df = pd.concat(
                [stats_df, pd.DataFrame([{"statistic": "unit", "value": unit}])],
                ignore_index=True
            )

        # （可选）总体非数值比例
        summary_df = pd.DataFrame({
            "statistic": ["non_numeric_total_ratio"],
            "value": [round(nonnum_mask.mean(), 6)]
        })

        feature_analysis[col] = {
            "stats": stats_df,
            "summary": summary_df,
            "non_numeric_categories": nonnum_df
        }

    else:
        # 非数值特征：类别占比（原样）
        # 特殊处理 Incubation temperature (℃)
        if col == "Incubation temperature (℃)":
            s = s.apply(
                lambda x: f"{x:.1f}" if isinstance(x, (int, float, np.number)) and not pd.isna(x) else x
            )
            
        # 特殊处理 Centrifugation repetitions
        if col == "Centrifugation repetitions":
            def normalize(x):
                try:
                    # NaN 直接返回
                    if pd.isna(x):
                        return x

                    v = float(x)  # 尝试转成数值

                    # 如果是整数（如 1, 1.0, "2"）
                    if v.is_integer():
                        return int(v)
                    else:
                        return round(v, 1)

                except (ValueError, TypeError):
                    # 不能转成数值的，原样返回
                    return x

            s = s.apply(normalize)
               
        vc = s.value_counts(normalize=True, dropna=False)
        df_cat = vc.reset_index()
        df_cat.columns = ["category", "proportion"]
        df_cat["percentage"] = (df_cat["proportion"] * 100).round(2)

        feature_analysis[col] = {"categorical": df_cat}

# 写入 Excel：数值特征一个sheet多块；类别特征一块
with pd.ExcelWriter("data_info_all.xlsx", engine="openpyxl") as writer:
    for feature_name, data in feature_analysis.items():
        sheet = str(feature_name)[0:31]

        # ====== 先写第一行：Feature: xxx ======
        # 写到 A1（Excel 行列从 1 开始）
        # pandas 写 Excel 是从 row=0 开始，所以 startrow=0 表示第一行
        pd.DataFrame([f"Feature: {feature_name}"]).to_excel(
            writer,
            sheet_name=sheet,
            index=False,
            header=False,
            startrow=0
        )

        # 数据从第 3 行开始写（留一行空行更好看）
        start_row = 2

        # ====== categorical 情况 ======
        if "categorical" in data:
            data["categorical"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )

        # ====== stats + summary + non_numeric_categories 情况 ======
        else:
            data["stats"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )

            start_row += len(data["stats"]) + 2
            data["summary"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )

            start_row += len(data["summary"]) + 3
            data["non_numeric_categories"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )
            
pd.reset_option("display.max_rows")











# %%
df = df_non_fill.copy()

# %%
threshold = 1e-6

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
df_plasma_human_high['Protein source organism'].unique()

# %%
df_plasma_human_high['Incubation protein source'].unique()

# %%
pd.set_option("display.max_rows", None)

# 提前确定的“应该是数值”的特征名列表
numeric_features = [
     "Primary size (nm)", "DLS size in water (nm)", "Zeta potential in water (mV)", 
     "PdI in water", "DLS size in dispersion medium (nm)", "Zeta potential in dispersion medium (mV)",
     "PdI in dispersion medium", "NM concentration", "Protein source concentration",
     "Incubation time (h)",  "Centrifugation speed",
     "Centrifugation time (min)", 
]


feature_analysis = {}

for col in df_non_fill.columns[1:38]:
    s = df_non_fill[col]

    if col in numeric_features:
        # -------- 新增：按列做清洗 + unit --------
        unit = None
        s_for_numeric = s

        if col == "NM concentration":
            # 去掉可能存在的 mg/L（大小写、空格）
            # 用 pandas StringDtype，保留缺失值为 <NA>
            s_for_numeric = (
                s.astype("string")
                 .str.replace("mg/L", "", regex=False)
                 .str.replace("MG/L", "", regex=False)
                 .str.replace("mg/l", "", regex=False)
                 .str.replace("MG/l", "", regex=False)
                 .str.strip()
            )
            unit = "mg/L"

        if col == "Protein source concentration":
            # 去掉可能存在的 %
            s_for_numeric = (
                s.astype("string")
                 .str.replace("%", "", regex=False)
                 .str.strip()
            )
            unit = "%"

        if col == "Centrifugation speed":
            # 去掉可能存在的 %
            s_for_numeric = (
                s.astype("string")
                 .str.replace("g", "", regex=False)
                 .str.strip()
            )
            unit = "g"

        # 1) 尝试整体转数值
        numeric = pd.to_numeric(s_for_numeric, errors="coerce")

        # 2) 转不成功的（含原本缺失 NaN）——按原始值逐类统计
        nonnum_mask = numeric.isna()
        nonnum_series = s[nonnum_mask]  # 这里按“原始值”分组统计（保留 mg/L、% 原样也可以对账）

        nonnum_counts = nonnum_series.value_counts(dropna=False)
        nonnum_df = nonnum_counts.reset_index()
        nonnum_df.columns = ["category", "count"]
        nonnum_df["proportion"] = (nonnum_df["count"] / len(s)).round(6)
        nonnum_df["percentage"] = (nonnum_df["proportion"] * 100).round(2)

        # 3) 数值统计（只对成功转数值的部分）
        numeric_valid = numeric.dropna()
        if len(numeric_valid) > 0:
            desc = numeric_valid.describe(percentiles=[0.25, 0.5, 0.75])
            stats_df = pd.DataFrame({
                "statistic": ["min", "max", "mean", "q25", "q50", "q75"],
                "value": [
                    desc["min"],
                    desc["max"],
                    desc["mean"],
                    desc["25%"],
                    desc["50%"],
                    desc["75%"],
                ]
            })
        else:
            stats_df = pd.DataFrame({"statistic": ["no_numeric_values"], "value": ["TRUE"]})

        # 4) 新增：unit 行（放在 stats 下面）
        if unit is not None:
            stats_df = pd.concat(
                [stats_df, pd.DataFrame([{"statistic": "unit", "value": unit}])],
                ignore_index=True
            )

        # （可选）总体非数值比例
        summary_df = pd.DataFrame({
            "statistic": ["non_numeric_total_ratio"],
            "value": [round(nonnum_mask.mean(), 6)]
        })

        feature_analysis[col] = {
            "stats": stats_df,
            "summary": summary_df,
            "non_numeric_categories": nonnum_df
        }

    else:
        # 非数值特征：类别占比（原样）
        # 特殊处理 Incubation temperature (℃)
        if col == "Incubation temperature (℃)":
            s = s.apply(
                lambda x: f"{x:.1f}" if isinstance(x, (int, float, np.number)) and not pd.isna(x) else x
            )
            
        # 特殊处理 Centrifugation repetitions
        if col == "Centrifugation repetitions":
            def normalize(x):
                try:
                    # NaN 直接返回
                    if pd.isna(x):
                        return x

                    v = float(x)  # 尝试转成数值

                    # 如果是整数（如 1, 1.0, "2"）
                    if v.is_integer():
                        return int(v)
                    else:
                        return round(v, 1)

                except (ValueError, TypeError):
                    # 不能转成数值的，原样返回
                    return x

            s = s.apply(normalize)
               
        vc = s.value_counts(normalize=True, dropna=False)
        df_cat = vc.reset_index()
        df_cat.columns = ["category", "proportion"]
        df_cat["percentage"] = (df_cat["proportion"] * 100).round(2)

        feature_analysis[col] = {"categorical": df_cat}

# 写入 Excel：数值特征一个sheet多块；类别特征一块
with pd.ExcelWriter("data_info_human_plasma.xlsx", engine="openpyxl") as writer:
    for feature_name, data in feature_analysis.items():
        sheet = str(feature_name)[0:31]

        # ====== 先写第一行：Feature: xxx ======
        # 写到 A1（Excel 行列从 1 开始）
        # pandas 写 Excel 是从 row=0 开始，所以 startrow=0 表示第一行
        pd.DataFrame([f"Feature: {feature_name}"]).to_excel(
            writer,
            sheet_name=sheet,
            index=False,
            header=False,
            startrow=0
        )

        # 数据从第 3 行开始写（留一行空行更好看）
        start_row = 2

        # ====== categorical 情况 ======
        if "categorical" in data:
            data["categorical"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )

        # ====== stats + summary + non_numeric_categories 情况 ======
        else:
            data["stats"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )

            start_row += len(data["stats"]) + 2
            data["summary"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )

            start_row += len(data["summary"]) + 3
            data["non_numeric_categories"].to_excel(
                writer,
                sheet_name=sheet,
                index=False,
                startrow=start_row
            )
            
pd.reset_option("display.max_rows")



# %%