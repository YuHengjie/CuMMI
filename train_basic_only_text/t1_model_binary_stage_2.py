# %%
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import boxcox
from datetime import datetime 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import sys
import argparse

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机数种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     
    torch.cuda.manual_seed(seed)                

set_seed(42)

# %%
# ================================
# Step 2: 构建 Dataset
# ================================
class EmbeddingPairDataset(Dataset):
    def __init__(self, df, text_embed_data, pro_esm_dict, pro_esmfold_dict):
        self.df = df.reset_index(drop=True)
        self.text_embed_data = text_embed_data
        self.pro_esm_dict = pro_esm_dict
        self.pro_esmfold_dict = pro_esmfold_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text_embed = self.text_embed_data[int(row["x_index"])]
        
        # Fetch the protein embedding
        pro_esm_embed = self.pro_esm_dict[row["Accession"]]
        pro_esmfold_embed = self.pro_esmfold_dict[row["Accession"]]
        
        rpa = row["Protein corona composition"]
        w = float(row.get("Overall data quality", 1.0))
        
        return torch.tensor(text_embed, dtype=torch.float32), \
               torch.tensor(pro_esm_embed, dtype=torch.float32), \
               torch.tensor(pro_esmfold_embed, dtype=torch.float32), \
               torch.tensor(rpa, dtype=torch.float32), \
               torch.tensor(w, dtype=torch.float32)

# %%
class CrossAttentionClassifierGated(nn.Module):
    def __init__(self, 
                 x_dim,              # 文本 / 纳米材料特征维度
                 pro_seq_dim,        # 蛋白序列特征维度（如 ESM2: 2560）
                 pro_str_dim,        # 蛋白结构特征维度（如 ESMFold: 384）
                 hidden_dim=1024, 
                 dropout=0.3):
        super().__init__()

        # --------- 文本侧：保持不变 ---------
        self.x_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --------- 注意力：保持不变 ---------
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout/2
        )
        
        # --------- 分类头：保持不变 ---------
        self.classifier  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, hidden_dim//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//16, 1)  # 二分类 logits（配合 BCEWithLogitsLoss）
        )
 
        # 初始化参数 
        self._init_weights()
 
    def _init_weights(self):
        # 原本的 x_mlp / pro_mlp / classifier 里的 Linear
        for module in [self.x_mlp, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.1)


    def forward(self, x_embed, pro_seq_embed, pro_str_embed):
        """
        x_embed       : [B, x_dim]
        pro_seq_embed : [B, pro_seq_dim]  (ESM2 输出等)
        pro_str_embed : [B, pro_str_dim]  (ESMFold 或结构特征)
        """

        # --------- 文本侧：保持不变 ---------
        x_feat = self.x_mlp(x_embed)        # [B, hidden_dim]
        
        # --------- 交叉注意力：保持不变 ---------
        x_tok   = x_feat.unsqueeze(1)   # [B, 1, hidden_dim]
        
        # 保留其他不变，减少代码修改，只使用文本模态
        attn_out, _ = self.attn(
            query=x_tok,
            key=x_tok,
            value=x_tok,
            need_weights=False
        )
        
        fused_feature = x_tok + attn_out          # [B, 1, hidden_dim]
        fused_feature = fused_feature.squeeze(1)  # [B, hidden_dim]
        
        # --------- 分类输出：保持不变 ---------
        logits = self.classifier(fused_feature).squeeze(-1)   # [B]

        # 返回 logits + gate，方便后面分析 gate 的使用情况
        return logits, None


# %%
@torch.no_grad()
def _eval_cls(logits_all, y_all, sample_weight=None, thresh=0.5):
    probs = torch.sigmoid(torch.tensor(logits_all)).numpy()
    y_true = torch.tensor(y_all).numpy().astype(int)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, probs, sample_weight=sw)
        aupr  = average_precision_score(y_true, probs, sample_weight=sw)
    else:
        auroc, aupr = np.nan, np.nan

    y_pred = (probs >= thresh).astype(int)
    f1  = f1_score(y_true, y_pred, sample_weight=sw, zero_division=0)
    acc = accuracy_score(y_true, y_pred, sample_weight=sw)
    return auroc, aupr, f1, acc, probs

def find_best_threshold(probs, y_true, sample_weight=None):
    grid = np.linspace(0.05, 0.95, 19)
    y_true = np.asarray(y_true).astype(int)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    best_t, best_f1 = 0.5, -1
    for t in grid:
        f1 = f1_score(y_true, (probs >= t).astype(int), sample_weight=sw, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def _validate_on_loader(model, loader, device, pos_weight=None):
    model.eval()
    val_loss = 0.0
    logits_all, y_all, w_all = [], [], []

    with torch.no_grad():
        for text, pro_seq, pro_str, y, w in loader:
            text = text.to(device, non_blocking=True)
            pro_seq = pro_seq.to(device, non_blocking=True)
            pro_str = pro_str.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            logits, _ = model(text, pro_seq, pro_str)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, weight=w, pos_weight=pos_weight, reduction='mean'
            )
            val_loss += loss.item()

            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())
            w_all.append(w.detach().cpu())

    avg_val_loss = val_loss / max(1, len(loader))
    logits_all = torch.cat(logits_all).numpy()
    y_all      = torch.cat(y_all).numpy().astype(int)
    w_all      = torch.cat(w_all).numpy().astype(float)

    auroc, aupr, f1, acc, probs = _eval_cls(logits_all, y_all, sample_weight=w_all, thresh=0.5)
    best_t, best_f1 = find_best_threshold(probs, y_all, sample_weight=w_all)

    return {
        'loss': avg_val_loss,
        'auroc': auroc, 'aupr': aupr, 'f1@0.5': f1, 'acc@0.5': acc,
        'best_t': best_t, 'best_f1': best_f1
    }

def train_and_validate_cls(model, train_loader, val_in_loader, val_out_loader,
                           epochs=100, lr=1e-4, log_interval=1,
                           pos_weight=None, save_dir='./train_result_cls'):
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.3*lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=3, min_lr=lr*0.01, cooldown=1
    )

    best_avg_aupr = -1.0
    history = {
        'train_loss':[],
        'val_in_loss':[],  'val_in_auroc':[],  'val_in_aupr':[],  'val_in_f1':[],  'val_in_acc':[],  'val_in_best_t':[],
        'val_out_loss':[], 'val_out_auroc':[], 'val_out_aupr':[], 'val_out_f1':[], 'val_out_acc':[], 'val_out_best_t':[],
        'avg_aupr':[], 'lr':[]
    }

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for text, pro_seq, pro_str, y, w in train_loader:
            text = text.to(device, non_blocking=True)
            pro_seq = pro_seq.to(device, non_blocking=True)
            pro_str = pro_str.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(text, pro_seq, pro_str)
            
            # 逐样本权重 + 正例权重（pos_weight）
            loss = F.binary_cross_entropy_with_logits(
                logits, y, weight=w, pos_weight=pos_weight, reduction='mean'
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ---- validate on both sets ----
        res_in  = _validate_on_loader(model, val_in_loader,  device, pos_weight=pos_weight)
        res_out = _validate_on_loader(model, val_out_loader,  device, pos_weight=pos_weight)
        avg_aupr = np.nanmean([res_in['aupr'], res_out['aupr']])  # 保存/调度看均值

        # ---- log ----
        cur_lr = scheduler.optimizer.param_groups[0]['lr']
        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch}/{epochs} | train_loss {avg_train_loss:.4f} \n"
                f"[IN]  val_loss {res_in['loss']:.4f}  AUROC {res_in['auroc']:.4f}  AUPRC {res_in['aupr']:.4f}  "
                f"F1@0.5 {res_in['f1@0.5']:.4f}  best_t {res_in['best_t']:.2f}  bestF1 {res_in['best_f1']:.4f} \n"
                f"[OUT] val_loss {res_out['loss']:.4f} AUROC {res_out['auroc']:.4f} AUPRC {res_out['aupr']:.4f} "
                f"F1@0.5 {res_out['f1@0.5']:.4f} best_t {res_out['best_t']:.2f} bestF1 {res_out['best_f1']:.4f} \n"
                f"AVG_AUPR {avg_aupr:.4f} | lr {cur_lr:.6f}"
            )

        # ---- history ----
        history['train_loss'].append(avg_train_loss)
        history['val_in_loss'].append(res_in['loss'])
        history['val_in_auroc'].append(res_in['auroc']);  history['val_in_aupr'].append(res_in['aupr'])
        history['val_in_f1'].append(res_in['f1@0.5']);    history['val_in_acc'].append(res_in['acc@0.5'])
        history['val_in_best_t'].append(res_in['best_t'])

        history['val_out_loss'].append(res_out['loss'])
        history['val_out_auroc'].append(res_out['auroc']); history['val_out_aupr'].append(res_out['aupr'])
        history['val_out_f1'].append(res_out['f1@0.5']);   history['val_out_acc'].append(res_out['acc@0.5'])
        history['val_out_best_t'].append(res_out['best_t'])

        history['avg_aupr'].append(avg_aupr)
        history['lr'].append(cur_lr)

        # ---- save best by average AUPR ----
        if avg_aupr > best_avg_aupr:
            best_avg_aupr = avg_aupr
            torch.save(model.state_dict(), os.path.join(save_dir, "saved_model_stage_2.pt"))
            print(f"✅ Saved best (AVG_AUPR={best_avg_aupr:.4f}) at epoch {epoch}")

        # ---- scheduler on average AUPR ----
        scheduler.step(avg_aupr)

    # save history
    np.save(os.path.join(save_dir, 'history_stage_2.npy'), history, allow_pickle=True)
    print(f"Training complete. Best AVG_AUPR: {best_avg_aupr:.4f}")
    return history


# %%


# %%
def print_model_params_count(model):
    # 遍历模型的每一个子模块
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if num_params > 0:  # 只打印有参数的模块
            print(f"Layer: {name}, Number of parameters: {num_params}")

# %%
# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 2 Model Training Script")
    # 可配置参数
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1.5e-4, help="Initial learning rate")
    parser.add_argument('--save_dir', type=str, default='./output/basic', help="Directory to save model and history")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loader workers")
    
    # 【新增参数】用于指定加载模型的阶段编号 (1, 2, 3, 4, 或 5)
    parser.add_argument('--load_stage', type=int, default=1, 
                        help="Stage number (N) of the model checkpoint to load (e.g., 1 loads saved_model_stage_1.pt).")
    
    # 检查是否在 Jupyter/IPython 环境中
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        # 如果在 Jupyter 中，则解析一个空列表，使用默认参数
        print("Running in Jupyter/IPython environment, using default arguments.")
        args = parser.parse_args([])
    else:
        # 否则，解析命令行参数
        args = parser.parse_args()
        
    # --- 4.1 数据加载 ---

    # 读取第一个文件
    with open("../protein_embedding/protein_embeddings_all_esm.pkl", "rb") as f:
        pro_esm_dict = pickle.load(f)
        
    # 检查第一个条目的数组维度
    first_key = next(iter(pro_esm_dict))  # 获取第一个键
    array_shape = pro_esm_dict[first_key].shape
    print(f"Array shape (pro_esm_dict) for {first_key}: {array_shape}")

    # 读取第一个文件
    with open("../protein_embedding/protein_embeddings_all_esmfold.pkl", "rb") as f:
        pro_esmfold_dict = pickle.load(f)
        
    # 检查第一个条目的数组维度
    first_key = next(iter(pro_esmfold_dict))  # 获取第一个键
    array_shape = pro_esmfold_dict[first_key].shape
    print(f"Array shape (pro_esmfold_dict) for {first_key}: {array_shape}")

    text_embed_data = np.load("../text_embedding/text_embeddings_nonfill.npy")  # 替换为文件路径
    print(f"Shape for text embedding: {text_embed_data.shape}")

    train_df = pd.read_csv("data/basic_serum_human_plasma_low_addhigh100_train.csv",keep_default_na=False, na_values=[''])
    val_in_df = pd.read_csv("data/basic_serum_human_plasma_low_addhigh100_val.csv",keep_default_na=False, na_values=[''])
    val_out_df = pd.read_csv("data/basic_plasma_human_high_val.csv",keep_default_na=False, na_values=[''])

    # --- 4.2 数据集和 DataLoader ---
    # Create datasets with the modified class
    train_dataset = EmbeddingPairDataset(train_df, text_embed_data, pro_esm_dict, pro_esmfold_dict)
    val_in_dataset = EmbeddingPairDataset(val_in_df, text_embed_data, pro_esm_dict, pro_esmfold_dict)
    val_out_dataset = EmbeddingPairDataset(val_out_df, text_embed_data, pro_esm_dict, pro_esmfold_dict)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_in_loader = DataLoader(val_in_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_out_loader = DataLoader(val_out_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- 4.3 模型初始化 ---
    # 获取维度
    x_dim = text_embed_data.shape[1]
    pro_esm_dim = next(iter(pro_esm_dict.values())).shape[0]
    pro_esmfold_dim = next(iter(pro_esmfold_dict.values())).shape[0]

    stage_ckpt = os.path.join(
                    args.save_dir, 
                    f'saved_model_stage_{args.load_stage}.pt'
                    )  # 保存的路径
    model = CrossAttentionClassifierGated(x_dim, pro_esm_dim, pro_esmfold_dim).to(device)
    model.load_state_dict(torch.load(stage_ckpt, map_location=device))
    print(f"Loaded {args.load_stage} best weights.")
    # 调用函数打印每一层的参数量
    print_model_params_count(model)
    
    # --- 4.4 计算 pos_weight ---
    # 统计类别分布 
    class_counts = train_df['Protein corona composition'].value_counts()

    # 获取具体数值 
    num_neg = class_counts.get(0,  0)  # 负样本数（类别0）
    num_pos = class_counts.get(1,  0)  # 正样本数（类别1）

    # 计算比例
    pos_weight_val = 2 * num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}, pos_weight: {pos_weight_val:.4f}")
    
    # --- 4.5 启动训练 ---
    print('\n***********Start Training Stage-2***********')
    history = train_and_validate_cls(
        model, train_loader, val_in_loader, val_out_loader,
        epochs=args.epochs, 
        lr=args.lr,
        pos_weight=pos_weight,
        save_dir=args.save_dir
    )
    
# %%