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
# ★設定項目：4モデルのパスをそれぞれ指定してください
# ==========================================
TARGET_ATTRIBUTE = "Microphone"
# 比較する送信パッチ数（全体196パッチのうち、何個残すか）
PATCH_COUNTS = [1, 2, 5, 10, 20, 50, 100, 150, 196] 


# 論文用：統一カラー＆マーカー設定
MODEL_STYLES = {
    "MAE":  {"color": "#1f77b4", "marker": "o"},
    "MOCO": {"color": "#ff7f0e", "marker": "s"},
    "BEIT": {"color": "#2ca02c", "marker": "^"},
    "DINO": {"color": "#d62728", "marker": "D"}
}


CONFIG_MAP = {
    "MAE" : {
    "stats_path":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_oid/for_dense_train_50k_each_2_run_11",
    "image_list":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    },

    "MOCO" : {
    "stats_path":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_moco/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    },

    "DINO" : 
    {
    "stats_path":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_dino/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    },

    "BEIT" : {
    "stats_path":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_beit/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
    }
}

# 保存先ディレクトリ
SAVE_DIR = f"./data/analysis_comparison_ablations/mae11_moco1_dino1_beit1/{TARGET_ATTRIBUTE}/applied_results_compression"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

def parse_best_info(path):
    if not os.path.exists(path): return None, None
    with open(path, 'r') as f:
        m = re.search(r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)", f.read())
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def load_backbone(m_type):
    m_type_upper = m_type.upper()
    try:
        if m_type_upper == "MAE": return timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE).eval()
        elif m_type_upper == "DINO": return timm.create_model("vit_base_patch16_224.dino", pretrained=True).to(DEVICE).eval()
        elif m_type_upper == "BEIT": return timm.create_model("beit_base_patch16_224", pretrained=True).to(DEVICE).eval()
        elif m_type_upper == "MOCO":
            model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
            url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            cp = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
            sd = cp.get('state_dict', cp.get('model', cp))
            nsd = {k.replace("module.base_encoder.", ""): v for k, v in sd.items() if k.startswith("module.base_encoder.")}
            model.load_state_dict(nsd, strict=False); return model.eval()
        return None
    except: return None

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tr = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    plt.figure(figsize=(10, 7)); all_model_data = []

    for name, paths in CONFIG_MAP.items():
        layer, unit = parse_best_info(paths["stats_path"])
        model = load_backbone(name)
        if model is None or layer is None: continue
        
        sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
        sae_path = os.path.join(paths["weights_dir"], f"sae_layer_{layer}.pth")
        if not os.path.exists(sae_path): continue
        sae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
        sae.eval()

        with open(paths["image_list"], 'r') as f:
            img_paths = [l.strip() for l in f.readlines() if l.strip()][:9]

        activations = {}
        handle = model.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: activations.update({'f': o.detach()}))
        compression_curves = []

        for p in tqdm(img_paths, desc=f"Compression Analysis {name}"):
            try:
                img_t = tr(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    model(img_t)
                    _, f = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                    f_patch = f.view(196, -1)[:, unit]
                    sorted_idx = torch.argsort(f_patch, descending=True).cpu().numpy()
                    base_act = f_patch.max().item() + 1e-8
                
                img_curve = []
                for count in PATCH_COUNTS:
                    reduced_img = torch.zeros_like(img_t)
                    for idx in sorted_idx[:count]:
                        gy, gx = divmod(idx, 14)
                        reduced_img[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16] = img_t[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16]
                    with torch.no_grad():
                        model(reduced_img)
                        _, f_new = sae(activations['f'][:, 1:, :].reshape(-1, 768))
                        img_curve.append(f_new.view(196, -1)[:, unit].max().item() / base_act * 100)
                compression_curves.append(img_curve)
            except: continue

        handle.remove()
        if not compression_curves: continue
        avg_curve = np.mean(compression_curves, axis=0)
        
        # ★統一されたスタイル（色・点）を使用
        style = MODEL_STYLES.get(name.upper(), {"color": None, "marker": "o"})
        plt.plot(PATCH_COUNTS, avg_curve, label=name, color=style["color"], marker=style["marker"], linewidth=2, markersize=6)
        
        res_entry = {"Model": name}
        for idx, count in enumerate(PATCH_COUNTS): res_entry[f"Patch_{count}_%"] = round(avg_curve[idx], 2)
        all_model_data.append(res_entry)
        del model, sae

    plt.title(f"Application Result: Semantic Data Reduction for '{TARGET_ATTRIBUTE}'", fontsize=14)
    plt.xlabel("Number of Patches Transmitted (Out of 196)"); plt.ylabel("Recognition Confidence (%)")
    plt.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label="80% Threshold")
    plt.ylim(0, 105); plt.legend(); plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(SAVE_DIR, f"compression_performance_{TARGET_ATTRIBUTE}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    pd.DataFrame(all_model_data).to_csv(os.path.join(SAVE_DIR, f"compression_report_{TARGET_ATTRIBUTE}.csv"), index=False)
    print(f"Results saved to: {save_path}")

if __name__ == "__main__": main()