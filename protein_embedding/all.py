# %%
# %%
import pickle

# %%
# 读取第一个文件
with open("protein_embeddings_all.pkl", "rb") as f:
    dict1 = pickle.load(f)

# %%
# 读取第二个文件
with open("protein_embeddings_unseen.pkl", "rb") as f:
    dict2 = pickle.load(f)

# %%
# 合并两个字典
# 注意：如果两个字典有相同的键，后面的会覆盖前面的
merged_dict = {**dict1, **dict2}

# 保存合并后的字典
with open("protein_embeddings_all_esm.pkl", "wb") as f:
    pickle.dump(merged_dict, f)




# %%
# 读取第一个文件
with open("esmfold_protein_embeddings.pkl", "rb") as f:
    dict1 = pickle.load(f)

# %%
# 读取第二个文件
with open("esmfold_protein_embeddings_unseen.pkl", "rb") as f:
    dict2 = pickle.load(f)

# %%
# 合并两个字典
# 注意：如果两个字典有相同的键，后面的会覆盖前面的
merged_dict = {**dict1, **dict2}

# 保存合并后的字典
with open("protein_embeddings_all_esmfold.pkl", "wb") as f:
    pickle.dump(merged_dict, f)

# %%
