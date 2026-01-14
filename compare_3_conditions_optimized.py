# 最適な閾値(analize_threshold_distribution.pyで保存したcsvファイル)に基づきSAE推定
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
import random
import glob
import hashlib

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
TARGET_ATTRIBUTE = "Microphone"

# 評価対象モデル
# ★表記を DINO v1, MoCo v3 に変更
MODELS_TO_EVALUATE = ["MAE", "MoCo v3", "BEiT", "DINO v1"] 

NUM_PATCHES_TO_KEEP = 20  
# ★修正点: 評価に使う枚数。キャリブレーション(200枚)との重複を避ける。
# フォルダ内の 201枚目 から 2000枚目 までをテストに使用します。
TEST_START_INDEX = 200  
NUM_TEST_PER_CLASS = 1800 

RANDOM_SEED = 42
PRECISION = 4  # 有効数字（小数点以下の桁数）を統一するための定数

# 閾値CSVのパス
THRESHOLD_CSV_PATH = f"./data/final_applied_results/threshold_analysis/{TARGET_ATTRIBUTE}/optimal_thresholds_{TARGET_ATTRIBUTE}.csv"

# データセットのルート
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

SAVE_DIR = f"./data/final_thesis_results/optimized_classification_separated/{TARGET_ATTRIBUTE}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONDITION_COLORS = {'Full': '#4daf4a', 'SAE-Guided': '#377eb8', 'Random': '#e41a1c'}


# ==========================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_stable_seed(path_string):
    return int(hashlib.md5(path_string.encode()).hexdigest(), 16) % (10**8)

def save_individual_plot(name, results_list):
    """特定のモデル単独のグラフを保存する（不透明設定）"""
    df_single = pd.DataFrame([r for r in results_list if r['Model'] == name])
    if df_single.empty: return
    fig, ax = plt.subplots(figsize=(7, 7))
    df_single['Condition'] = pd.Categorical(df_single['Condition'], categories=["Full", "SAE-Guided", "Random"], ordered=True)
    df_single = df_single.sort_values('Condition')
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # 小数点以下桁数をPRECISIONに合わせて表示
    bars = ax.bar(df_single['Condition'], df_single['Accuracy'], 
                   color=[CONDITION_COLORS[c] for c in df_single['Condition']], 
                   alpha=1.0, edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.{PRECISION}f}", 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(f"Accuracy: {name}\nTarget: {TARGET_ATTRIBUTE}", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy Score"); ax.set_ylim(0, 1.1)
    save_path = os.path.join(SAVE_DIR, f"opt_accuracy_{name}_{TARGET_ATTRIBUTE}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()

def main():
    set_seed(RANDOM_SEED); os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(THRESHOLD_CSV_PATH):
        print(f" [Error] CSV not found: {THRESHOLD_CSV_PATH}")
        return
    csv_configs = pd.read_csv(THRESHOLD_CSV_PATH).set_index('Model').to_dict('index')

    pos_paths = sorted(glob.glob(os.path.join(POS_DIR, "*.jpg")))[TEST_START_INDEX:TEST_START_INDEX+NUM_TEST_PER_CLASS]
    neg_paths = sorted(glob.glob(os.path.join(NEG_DIR, "*.jpg")))[TEST_START_INDEX:TEST_START_INDEX+NUM_TEST_PER_CLASS]
    test_data = [(p, 1) for p in pos_paths] + [(p, 0) for p in neg_paths]

    tr = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    all_results = []

    for name in MODELS_TO_EVALUATE:
        if name not in csv_configs: continue
        cfg = csv_configs[name]
        layer, unit = int(cfg['Layer']), int(cfg['Unit'])
        t_full = float(cfg['Threshold_Full'])
        t_mask = float(cfg['Threshold_Masked'])
        
        print(f"\n>>> Testing {name} (L{layer} U{unit})")
        
        if name == "MoCo v3":
            model = timm.create_model(CONFIG_MAP[name]["model_id"], pretrained=False).to(DEVICE)
            url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            cp = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
            model.load_state_dict({k.replace("module.base_encoder.", ""): v for k, v in cp['state_dict'].items() if k.startswith("module.base_encoder.")}, strict=False)
        else:
            model = timm.create_model(CONFIG_MAP[name]["model_id"], pretrained=True).to(DEVICE)
        model.eval()

        sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
        sae.load_state_dict(torch.load(os.path.join(CONFIG_MAP[name]["weights_dir"], f"sae_layer_{layer}.pth"), map_location=DEVICE))
        sae.eval()

        stats = {c: {"tp":0, "fp":0, "tn":0, "fn":0} for c in ["Full", "SAE-Guided", "Random"]}
        activations = {}
        handle = model.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: activations.update({'f': o.detach()}))

        for path, label in tqdm(test_data, desc=f" Testing {name}"):
            try:
                img_t = tr(Image.open(path).convert('RGB')).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    model(img_t)
                    _, f = sae(activations['f'][:, 1:, :].reshape(-1, 768)); f_p = f[:, unit]
                    pred_full = 1 if f_p.max().item() >= t_full else 0
                    sorted_idx = torch.argsort(f_p, descending=True).cpu().numpy()

                    mask_sae = torch.zeros_like(img_t)
                    for idx in sorted_idx[:NUM_PATCHES_TO_KEEP]:
                        gy, gx = divmod(idx, 14); mask_sae[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16] = img_t[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16]
                    model(mask_sae); _, f_s = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                    pred_sae = 1 if f_s[:, unit].max().item() >= t_mask else 0

                    random.seed(get_stable_seed(path)); rnd_idx = list(range(196)); random.shuffle(rnd_idx)
                    mask_rnd = torch.zeros_like(img_t)
                    for idx in rnd_idx[:NUM_PATCHES_TO_KEEP]:
                        gy, gx = divmod(idx, 14); mask_rnd[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16] = img_t[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16]
                    model(mask_rnd); _, f_r = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                    pred_rnd = 1 if f_r[:, unit].max().item() >= t_mask else 0

                def update_m(cond, pred):
                    if label==1 and pred==1: stats[cond]["tp"]+=1
                    elif label==1 and pred==0: stats[cond]["fn"]+=1
                    elif label==0 and pred==1: stats[cond]["fp"]+=1
                    elif label==0 and pred==0: stats[cond]["tn"]+=1
                update_m("Full", pred_full); update_m("SAE-Guided", pred_sae); update_m("Random", pred_rnd)
            except: continue

        handle.remove()
        
        # 精度と維持率、そして「Filtering Gain」を算出
        acc_vals = {}
        for cond in ["Full", "SAE-Guided", "Random"]:
            s = stats[cond]
            acc = (s["tp"] + s["tn"]) / (s["tp"] + s["tn"] + s["fp"] + s["fn"] + 1e-8)
            acc_vals[cond] = acc

        gain = acc_vals["SAE-Guided"] - acc_vals["Random"]
        
        current_model_results = []
        for cond in ["Full", "SAE-Guided", "Random"]:
            acc = acc_vals[cond]
            maintenance_rate = acc / (acc_vals["Full"] + 1e-8)
            res = {
                "Model": name, 
                "Condition": cond, 
                # PRECISION定数に基づいて小数点以下の桁数を統一
                "Accuracy": format(acc, f'.{PRECISION}f'), 
                "Maintenance_Rate": format(maintenance_rate, f'.{PRECISION}f'),
                "Filtering_Gain": format(gain, f'.{PRECISION}f')
            }
            all_results.append(res)
            current_model_results.append(res)


        # グラフ保存用関数の内部でも浮動小数点として扱うため再度変換
        plot_results = [{**r, "Accuracy": float(r["Accuracy"])} for r in current_model_results]
        save_individual_plot(name, plot_results); del model, sae

    # レポート作成
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(SAVE_DIR, f"opt_accuracy_summary_with_gain_{TARGET_ATTRIBUTE}.csv")
    df.to_csv(csv_path, index=False)
    
    # グラフ表示
    # Accuracyを数値に戻してプロット
    df_for_plot = df.copy()
    df_for_plot["Accuracy"] = df_for_plot["Accuracy"].astype(float)
    pivot_df = df_for_plot.pivot(index="Model", columns="Condition", values="Accuracy")[["Full", "SAE-Guided", "Random"]].reindex(MODELS_TO_EVALUATE)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_axisbelow(True); ax.grid(axis='y', linestyle='--', alpha=0.4)
    pivot_df.plot(kind="bar", ax=ax, color=[CONDITION_COLORS[c] for c in pivot_df.columns], alpha=1.0, edgecolor='black')
    plt.title(f"Accuracy with Double Calibration: {TARGET_ATTRIBUTE}", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy Score"); plt.ylim(0, 1.1); plt.xticks(rotation=0)
    plt.savefig(os.path.join(SAVE_DIR, f"opt_summary_chart_with_gain_{TARGET_ATTRIBUTE}.png"), dpi=300, bbox_inches='tight'); plt.close()
    
    print("\n" + "="*70 + "\n EVALUATION COMPLETE (Filtering Gain Analysis)\n" + "="*70)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()