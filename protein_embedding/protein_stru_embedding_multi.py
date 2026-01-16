# %%
import torch
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
import multiprocessing as mp
import pickle # 导入 pickle 库
import os # 导入 os 库，用于检查文件是否存在

# %%
# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================
# 测试显存占用
# ==============================
def test_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} 显存分配情况：")
        print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  保留: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

# test_gpu_memory()

# %%
# ==============================
# 分段函数（带 overlap）
# ==============================
max_len = 1024   # 每段最大长度
overlap = 64

def chunk_sequence(seq, chunk_size=max_len, overlap=overlap):
    """把序列切成若干段，每段有 overlap 个氨基酸与前一段重叠。"""
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    step = chunk_size - overlap
    chunks = [seq[i:i + chunk_size] for i in range(0, len(seq), step)]
    return chunks

# %%
# 定义全局保存路径
EMBEDDING_FILE = "esmfold_protein_embeddings.pkl"

def clean_sequence(seq):
    seq = seq.strip().upper()
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join([aa if aa in valid_aas else "X" for aa in seq])
    return cleaned

def compute_protein_embedding_single_gpu(seq, model, tokenizer, chunk_size, overlap, device):
    """单序列切片处理函数 (与你原代码的 compute_protein_embedding 相似)"""
    # ... (使用你原代码中的 compute_protein_embedding 逻辑，确保它只接受单个 seq)
    # 此处省略具体实现，沿用你原始代码中单序列的处理逻辑，无需 batching
    seq = clean_sequence(seq)
    chunks = chunk_sequence(seq, chunk_size, overlap)
    all_embeddings = []

    with torch.no_grad():
        for chunk in chunks:
            tokenized_input = tokenizer([chunk], return_tensors="pt", 
                                        add_special_tokens=False,padding=True,)["input_ids"].to(device)
            # 直接调用模型，因为没有 DataParallel
            output = model(tokenized_input) 
            # 确保 output["states"] 的 shape 是 [1, L, 384] 或类似
            last_layer = output["states"][-1, 0] 
            chunk_emb = last_layer.mean(dim=0)
            all_embeddings.append(chunk_emb.cpu().to(torch.float32))
            
            # 清理显存
            del tokenized_input, output, last_layer
            torch.cuda.empty_cache()

    all_embeddings = torch.stack(all_embeddings)
    protein_embedding = all_embeddings.mean(dim=0)
    return protein_embedding.cpu().numpy()

def gpu_worker(rank, df_subset, model_path, tokenizer_path, chunk_size, overlap, result_queue):
    """每个 GPU 进程执行的函数"""
    device = torch.device(f"cuda:{rank}")
    print(f"Worker {rank}: Loading model on {device}")
    
    # 加载模型和 tokenizer 到各自的 GPU
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EsmForProteinFolding.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    
    # 推理
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"GPU {rank} Progress"): # 增加子进程 tqdm
        accession = row["Accession"]
        seq = row["Sequence"]
        
        try:
            # 调用单序列处理函数
            emb = compute_protein_embedding_single_gpu(seq, model, tokenizer, chunk_size, overlap, device)
            result_queue.put((accession, emb))
        except Exception as e:
            print(f"Error processing {accession} on GPU {rank}: {e}")
            # 如果出错，仍然发送 None，主进程会忽略，但我们在这里仍然要继续
            result_queue.put((accession, None)) 
            
    # **🌟 关键修改：发送结束信号 🌟**
    result_queue.put(('SENTINEL', None)) 
            
    # 显式清理
    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"Worker {rank}: Finished.")
    
# ==============================
# 主进程：协调与保存
# ==============================
def run_parallel_inference(df, model_path, tokenizer_path, chunk_size, overlap, num_gpus):

    # Step A: 加载已有结果
    embedding_dict = {}
    if os.path.exists(EMBEDDING_FILE):
        try:
            with open(EMBEDDING_FILE, "rb") as f:
                embedding_dict = pickle.load(f)
            print(f"Loaded checkpoint with {len(embedding_dict)} embeddings.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            embedding_dict = {}

    processed_accessions = set(embedding_dict.keys())
    df_unprocessed = df[~df["Accession"].isin(processed_accessions)]

    if len(df_unprocessed) == 0:
        print("All sequences are already encoded.")
        return embedding_dict

    print(f"Total sequences to process: {len(df_unprocessed)}")

    # Step B: 切分数据
    num_gpus = min(num_gpus, len(df_unprocessed))
    df_splits = np.array_split(df_unprocessed, num_gpus)
    result_queue = mp.Queue()

    # Step C: 启动子进程
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, df_splits[rank], model_path, tokenizer_path, chunk_size, overlap, result_queue),
        )
        p.start()
        processes.append(p)

    # Step D: 主进程收集结果并保存
    active_processes = num_gpus
    SAVE_INTERVAL = max(100, num_gpus)
    pbar = tqdm(total=len(df_unprocessed), desc="Overall Progress")

    while active_processes > 0:
        try:
            accession, emb = result_queue.get(timeout=1)

            if accession == "SENTINEL":
                active_processes -= 1
                continue

            if emb is not None:
                embedding_dict[accession] = emb
                pbar.update(1)

                # 定期保存（原子写）
                if len(embedding_dict) % SAVE_INTERVAL == 0:
                    tmp_file = EMBEDDING_FILE + ".tmp"
                    with open(tmp_file, "wb") as f:
                        pickle.dump(embedding_dict, f)
                    os.replace(tmp_file, EMBEDDING_FILE)
                    pbar.set_postfix({"Saved": len(embedding_dict)})

        except Exception:
            pass  # 队列暂时为空

    for p in processes:
        p.join()

    # Step E: 最终保存
    tmp_file = EMBEDDING_FILE + ".tmp"
    with open(tmp_file, "wb") as f:
        pickle.dump(embedding_dict, f)
    os.replace(tmp_file, EMBEDDING_FILE)

    pbar.close()
    print(f"✅ Finished encoding. Total proteins: {len(embedding_dict)} saved to {EMBEDDING_FILE}")
    return embedding_dict


# %%
# =========================================================
# 主执行入口
# =========================================================
if __name__ == '__main__':
    # 1. 设置启动方法 (放在这里，防止重复设置)
    try:
        # force=True 确保设置生效
        mp.set_start_method('spawn', force=True) 
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Could not set start method: {e}")
        
    # 2. 全局参数和数据加载 (仅在主进程中执行一次)
    model_path = "/home/yuhengjie/pt_model/esmfold_v1"
    tokenizer_path = model_path
    num_gpus = 8
    max_len = 1024
    overlap = 64
    
    # 假设 df 已加载
    # ⚠️ 确保 pd.read_excel 也在 if __name__ == '__main__': 块内
    df = pd.read_excel("protein_seq_20250418.xlsx", index_col=0)

    # 3. 运行并行推理 (仅在主进程中执行一次)
    print(f"Starting parallel inference on {num_gpus} GPUs...")
    embedding_dict = run_parallel_inference(df, model_path, tokenizer_path, max_len, overlap, num_gpus)

    print(f"Total proteins encoded: {len(embedding_dict)}")
    
# %%