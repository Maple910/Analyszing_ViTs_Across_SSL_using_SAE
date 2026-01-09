import os
import re
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from sae_model import SparseAutoencoder
from tqdm import tqdm

# ==========================================
# ★設定項目：各モデルのパスを指定してください
# ==========================================
TARGET_ATTRIBUTE = "Microphone"

# 各モデルのパス設定 (run_all_model_ablations.py と同じ形式)
MAE_PATHS = {
    "stats_path":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_oid/for_dense_train_50k_each_2_run_11",
    "image_list":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

MOCO_PATHS = {
    "stats_path":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_moco/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

DINO_PATHS = {
    "stats_path":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_dino/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

BEIT_PATHS = {
    "stats_path":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_beit/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

# 消去するパッチの最大数
MAX_PATCHES_TO_DELETE = 15
SAVE_DIR = f"./data/analysis_comparison_ablations/mae11_moco1_dino1_beit1/{TARGET_ATTRIBUTE}/curves"

# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_unit(path):
    if not os.path.exists(path): return None, None
    with open(path, 'r') as f:
        m = re.search(r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)", f.read())
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def load_backbone(m_type):
    if m_type == "MAE":  return timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE).eval()
    if m_type == "DINO": return timm.create_model("vit_base_patch16_224.dino", pretrained=True).to(DEVICE).eval()
    if m_type == "BEiT": return timm.create_model("beit_base_patch16_224", pretrained=True).to(DEVICE).eval()
    if m_type == "MoCo":
        model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
        cp = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar", map_location=DEVICE)
        sd = cp.get('state_dict', cp.get('model', cp))
        nsd = {k.replace("module.base_encoder.", ""): v for k, v in sd.items() if k.startswith("module.base_encoder.")}
        model.load_state_dict(nsd, strict=False); return model.eval()

def get_deletion_curve(model, sae, img_tensor, layer, unit_id):
    """重要なパッチから順に消していき、活性の変化を記録する"""
    activations = {}
    handle = model.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: activations.update({'f': o.detach()}))
    
    with torch.no_grad():
        model(img_tensor)
        _, f = sae(activations['f'][:, 1:, :].reshape(-1, 768))
        f_patch = f.view(196, -1)[:, unit_id]
        
    # 活性が高い順にパッチインデックスを並べる
    sorted_indices = torch.argsort(f_patch, descending=True).cpu().numpy()
    
    curve = []
    temp_img = img_tensor.clone()
    
    for n in range(MAX_PATCHES_TO_DELETE + 1):
        with torch.no_grad():
            model(temp_img)
            _, f_new = sae(activations['f'][:, 1:, :].reshape(-1, 768))
            max_act = f_new.view(196, -1)[:, unit_id].max().item()
            curve.append(max_act)
        
        if n < MAX_PATCHES_TO_DELETE:
            idx = sorted_indices[n]
            gy, gx = divmod(idx, 14)
            temp_img[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16] = 0
            
    handle.remove()
    # 最初の値を100%として正規化
    base = curve[0] + 1e-8
    return [c / base * 100 for c in curve]

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tr = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    tasks = [("MAE", MAE_PATHS, "#1f77b4", "o"), ("MoCo", MOCO_PATHS, "#ff7f0e", "s"), 
             ("BEiT", BEIT_PATHS, "#2ca02c", "^"), ("DINO", DINO_PATHS, "#d62728", "D")]

    plt.figure(figsize=(10, 6))
    
    for name, paths, color, marker in tasks:
        print(f">>> Analyzing curve for {name}...")
        layer, unit = parse_unit(paths["stats_path"])
        if layer is None: continue
        
        model = load_backbone(name)
        sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
        sae.load_state_dict(torch.load(os.path.join(paths["weights_dir"], f"sae_layer_{layer}.pth"), map_location=DEVICE))
        sae.eval()

        with open(paths["image_list"], 'r') as f:
            img_paths = [l.strip() for l in f.readlines() if l.strip()][:5] # 上位5枚で平均を取る

        all_curves = []
        for p in img_paths:
            img_t = tr(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
            all_curves.append(get_deletion_curve(model, sae, img_t, layer, unit))
        
        avg_curve = np.mean(all_curves, axis=0)
        plt.plot(range(MAX_PATCHES_TO_DELETE + 1), avg_curve, label=name, color=color, marker=marker, linewidth=2)
        
        del model, sae

    plt.title(f"Activation Deletion Curve: '{TARGET_ATTRIBUTE}'", fontsize=15, fontweight='bold')
    plt.xlabel("Number of Most Active Patches Removed", fontsize=12)
    plt.ylabel("Relative Max Activation (%)", fontsize=12)
    plt.xticks(range(MAX_PATCHES_TO_DELETE + 1))
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    save_path = os.path.join(SAVE_DIR, f"ablation_curve_{TARGET_ATTRIBUTE}.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"\n[Success] 曲線プロットを保存しました: {save_path}")

if __name__ == "__main__":
    main()