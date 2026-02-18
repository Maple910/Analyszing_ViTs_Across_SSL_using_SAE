# パッチを順次削除していき維持率がどの程度下がっていくかを見る
# グラフ，csv作成

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

# ==========================================
# ★設定項目
# ==========================================
TARGET_ATTRIBUTE = "Microphone"
# AMR実験（compare_ablation_experiment_semantic.py）のステップと統一
# 0パッチ削除（100%状態）を基点として追加
DELETE_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 150, 196] 
NUM_IMAGES_FOR_STATS = 9  

MODEL_STYLES = {
    "MAE":      {"color": "#1f77b4", "marker": "o"},
    "MOCO V3":  {"color": "#ff7f0e", "marker": "s"},
    "BEIT":     {"color": "#2ca02c", "marker": "^"},
    "DINO V1":  {"color": "#d62728", "marker": "D"}
}

CONFIG_MAP = {
    "MAE" : {
        "stats_path":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_oid/for_dense_train_50k_each_2_run_11",
        "image_list":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    },
    "MoCo v3" : {
        "stats_path":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_moco/for_dense_train_50k_each_2_run_1",
        "image_list":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    },
    "BEiT" : {
        "stats_path":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_beit/for_dense_train_50k_each_2_run_1",
        "image_list":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    },
    "DINO v1" : {
        "stats_path":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
        "weights_dir":  f"./data/sae_weights_dino/for_dense_train_50k_each_2_run_1",
        "image_list":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    }
}

SAVE_DIR = f"./data/analysis_comparison_ablations/mae11_moco1_dino1_beit1/{TARGET_ATTRIBUTE}/report_aligned_steps"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_best_info(path):
    if not os.path.exists(path): return None, None
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)", content)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)

def load_backbone(m_type):
    m_type_upper = m_type.upper()
    try:
        if "MAE" in m_type_upper: return timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE).eval()
        elif "DINO" in m_type_upper: return timm.create_model("vit_base_patch16_224.dino", pretrained=True).to(DEVICE).eval()
        elif "BEIT" in m_type_upper: return timm.create_model("beit_base_patch16_224", pretrained=True).to(DEVICE).eval()
        elif "MOCO" in m_type_upper:
            model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
            url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
            sd = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            nsd = {k.replace("module.base_encoder.", ""): v for k, v in sd.items() if k.startswith("module.base_encoder.")}
            model.load_state_dict(nsd, strict=False)
            return model.eval()
        return None
    except: return None

def compute_auc(x_values, y_values):
    # np.trapzは不等間隔のxに対しても正しく面積を計算します
    return np.trapz(y_values, x_values) / (196 * 100)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tr = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    results_summary = []
    plt.figure(figsize=(10, 6))

    for name, paths in CONFIG_MAP.items():
        layer, unit = parse_best_info(paths["stats_path"])
        model = load_backbone(name)
        if model is None or layer is None: continue
            
        sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
        sae_path = os.path.join(paths["weights_dir"], f"sae_layer_{layer}.pth")
        if not os.path.exists(sae_path): continue
        sae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
        sae.eval()

        with open(paths["image_list"], 'r', encoding='utf-8') as f:
            img_paths = [l.strip() for l in f.readlines() if l.strip()][:NUM_IMAGES_FOR_STATS]

        model_curves = []
        activations = {}
        handle = model.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: activations.update({'f': o.detach()}))

        for p in tqdm(img_paths, desc=f"Aligned Ablation Analysis {name}"):
            try:
                img_t = tr(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    model(img_t)
                    _, f = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                    f_patch = f.view(196, -1)[:, unit]
                    sorted_idx = torch.argsort(f_patch, descending=True).cpu().numpy()
                    base_act = f_patch.max().item() + 1e-8

                img_curve = []
                for n in DELETE_STEPS:
                    temp_img = img_t.clone()
                    if n > 0:
                        # 上位nパッチを黒塗り
                        for idx in sorted_idx[:n]:
                            gy, gx = divmod(idx, 14)
                            temp_img[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16] = 0
                    
                    with torch.no_grad():
                        model(temp_img)
                        _, f_new = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                        val = f_new.view(196, -1)[:, unit].max().item()
                        img_curve.append(val / base_act * 100)
                model_curves.append(img_curve)
            except: continue

        handle.remove()
        if not model_curves: continue
            
        avg_curve = np.mean(model_curves, axis=0)
        auc_score = compute_auc(DELETE_STEPS, avg_curve)
        
        # 記録用
        res_entry = {"Model": name, "AUC": round(auc_score, 4)}
        for i, count in enumerate(DELETE_STEPS):
            res_entry[f"Del_{count}_act_%"] = round(avg_curve[i], 4)
        results_summary.append(res_entry)

        style = MODEL_STYLES.get(name.upper(), {"color": None, "marker": "o"})
        plt.plot(DELETE_STEPS, avg_curve, 
                 label=f"{name} (AUC: {auc_score:.4f})", 
                 color=style["color"], marker=style["marker"], markersize=5, linewidth=1.5)
        del model, sae

    plt.title(f"Ablation Curve (Aligned with AMR steps): {TARGET_ATTRIBUTE}", fontsize=14)
    plt.xlabel("Number of Most Active Patches Removed")
    plt.ylabel("Relative Max Activation (%)")
    plt.xlim(-5, 210); plt.ylim(0, 105)
    plt.xticks([0, 10, 20, 50, 100, 150, 196]) 
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(SAVE_DIR, f"robustness_curve_aligned_{TARGET_ATTRIBUTE}.png")
    plt.savefig(save_path, dpi=200)
    pd.DataFrame(results_summary).to_csv(os.path.join(SAVE_DIR, f"robustness_report_aligned_{TARGET_ATTRIBUTE}.csv"), index=False)
    print(f"Results saved to: {save_path}")

if __name__ == "__main__": main()