# SAE推定　最適な閾値計算
import os
import re
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from sae_model import SparseAutoencoder
from tqdm import tqdm
import glob
import random
# ==========================================
# 特徴一覧
"""
    "Person",
    "Car",
    "Guitar",
    "Table",
    "Mobile_phone",
    "Bird",
    "Sunglasses",
    "Tree",
    "Building",
    "Chair",
    "Microphone"
"""
# ==========================================

# ==========================================
# ★設定項目
# ==========================================
TARGET_ATTRIBUTE = "Chair"

# 閾値を決めるための「キャリブレーション用」の枚数（各クラス200枚ずつ、計400枚を使用）
NUM_CALIB_SAMPLES = 200 
NUM_PATCHES_TO_KEEP = 20 # 10%相当
RANDOM_SEED = 42

# データセットのルートパス
DATASET_ROOT = "./data/oid_dataset"
POS_DIR = os.path.join(DATASET_ROOT, TARGET_ATTRIBUTE, "positive")
NEG_DIR = os.path.join(DATASET_ROOT, TARGET_ATTRIBUTE, "negative")

# 各モデルの解析パス設定
# ★表記を DINO v1, MoCo v3 に変更
CONFIG_MAP = {
    "MAE": {
        "model_id": "vit_base_patch16_224.mae",
        "stats_path":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_oid/for_dense_train_50k_each_2_run_11"
    },
    "MoCo v3": {
        "model_id": "vit_base_patch16_224", 
        "stats_path":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_moco/for_dense_train_50k_each_2_run_1"
    },
    "DINO v1": {
        "model_id": "vit_base_patch16_224.dino",
        "stats_path":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_dino/for_dense_train_50k_each_2_run_1"
    },
    "BEiT": {
        "model_id": "beit_base_patch16_224",
        "stats_path":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_beit/for_dense_train_50k_each_2_run_1"
    }
}

SAVE_DIR = f"./data/final_applied_results/threshold_analysis/{TARGET_ATTRIBUTE}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_image_paths(directory, count):
    paths = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    return paths[:count]

def parse_best_unit(path):
    if not os.path.exists(path): return None, None
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.search(r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)", content)
    if not m: m = re.search(r"Layer\s+(\d+),\s+Unit\s+(\d+)", content, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def find_optimal_threshold(pos_scores, neg_scores):
    all_scores = sorted(list(set(pos_scores + neg_scores)))
    best_acc, best_t = 0, 0
    search_candidates = [all_scores[i] for i in np.linspace(0, len(all_scores)-1, min(500, len(all_scores)), dtype=int)] if all_scores else [0]
    for t in search_candidates:
        tp = sum(1 for s in pos_scores if s >= t)
        tn = sum(1 for s in neg_scores if s < t)
        acc = (tp + tn) / (len(pos_scores) + len(neg_scores) + 1e-8)
        if acc >= best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc

def save_dist_plot(pos_scores, neg_scores, threshold, model_name, condition, save_path):
    """分布図をプロットして保存するヘルパー関数"""
    plt.figure(figsize=(10, 6))
    plt.hist(pos_scores, bins=40, alpha=0.6, label="Positive", color="#1f77b4", density=True)
    plt.hist(neg_scores, bins=40, alpha=0.6, label="Negative", color="#d62728", density=True)
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f"Optimal Thresh: {threshold:.4f}")
    plt.title(f"SAE Activation Distribution: {model_name} ({condition})\nTarget: {TARGET_ATTRIBUTE}")
    plt.xlabel("Activation Intensity"); plt.ylabel("Density"); plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    set_seed(RANDOM_SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    tr = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pos_img_paths = get_image_paths(POS_DIR, NUM_CALIB_SAMPLES)
    neg_img_paths = get_image_paths(NEG_DIR, NUM_CALIB_SAMPLES)
    
    summary_results = []

    for name, config in CONFIG_MAP.items():
        print(f"\n>>> Analyzing {name} Activation Distribution (Double-Calibration)...")
        layer, unit = parse_best_unit(config["stats_path"])
        if layer is None: continue
            
        if name == "MoCo v3":
            model = timm.create_model(config["model_id"], pretrained=False).to(DEVICE)
            url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            cp = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
            model.load_state_dict({k.replace("module.base_encoder.", ""): v for k, v in cp['state_dict'].items() if k.startswith("module.base_encoder.")}, strict=False)
        else:
            model = timm.create_model(config["model_id"], pretrained=True).to(DEVICE)
        model.eval()

        sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
        sae.load_state_dict(torch.load(os.path.join(config["weights_dir"], f"sae_layer_{layer}.pth"), map_location=DEVICE))
        sae.eval()

        full_scores = {"pos": [], "neg": []}
        mask_scores = {"pos": [], "neg": []}
        activations = {}
        handle = model.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: activations.update({'f': o.detach()}))

        for label, paths in [("pos", pos_img_paths), ("neg", neg_img_paths)]:
            for p in tqdm(paths, desc=f" {name} {label}"):
                try:
                    img_t = tr(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        # Full画像での活性化
                        model(img_t)
                        _, f = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                        act_f = f[:, unit]
                        full_scores[label].append(act_f.max().item())
                        
                        # 10%削減（SAE-Guided）状態での活性化
                        sorted_idx = torch.argsort(act_f, descending=True).cpu().numpy()
                        mask_img = torch.zeros_like(img_t)
                        for idx in sorted_idx[:NUM_PATCHES_TO_KEEP]:
                            gy, gx = divmod(idx, 14)
                            mask_img[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16] = img_t[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16]
                        model(mask_img)
                        _, f_m = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                        mask_scores[label].append(f_m[:, unit].max().item())
                except: continue
        
        handle.remove()
        
        # それぞれの状態での最適閾値を算出
        t_full, acc_f = find_optimal_threshold(full_scores["pos"], full_scores["neg"])
        t_mask, acc_m = find_optimal_threshold(mask_scores["pos"], mask_scores["neg"])
        
        summary_results.append({
            "Model": name, "Layer": layer, "Unit": unit, 
            "Threshold_Full": round(t_full, 4), 
            "Threshold_Masked": round(t_mask, 4),
            "Acc_Full": round(acc_f, 3),
            "Acc_Masked": round(acc_m, 3)
        })

        # --- グラフ保存 (Full と Masked を個別に作成) ---
        save_dist_plot(full_scores["pos"], full_scores["neg"], t_full, name, "Full", 
                       os.path.join(SAVE_DIR, f"calibration_dist_Full_{name}.png"))
        
        save_dist_plot(mask_scores["pos"], mask_scores["neg"], t_mask, name, "Masked_10pct", 
                       os.path.join(SAVE_DIR, f"calibration_dist_Masked_{name}.png"))

        del model, sae

    # CSV保存
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(SAVE_DIR, f"optimal_thresholds_{TARGET_ATTRIBUTE}.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*60 + "\n DOUBLE CALIBRATION COMPLETE & PLOTS SAVED\n" + "="*60)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()