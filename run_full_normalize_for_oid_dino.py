# 指定した特徴群（TRAGET_ATTRIBUTES）に対して可視化を行う（DINO用）

import os
import subprocess
import re
import time
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# 必要なモジュールをインポート
from sae_model import SparseAutoencoder
from data_loader_oid import get_openimages_attribute_loaders

# 定数読み込み用 (DINO用)
import config_dino as cfg_dino

# ==========================================
# ★実験したい属性リスト
# ==========================================
TARGET_ATTRIBUTES = [    
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
]
# ==========================================

# ファイルパス設定
CONFIG_PATH = "config_dino.py"
SCRIPT_COMPARE = "compare_attribute_normalization_oid_dino.py"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def update_config_file(file_path, new_attribute):
    """Configファイルの TARGET_ATTRIBUTE を書き換える"""
    if not os.path.exists(file_path):
        print(f"[ERROR] Config file not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    updated = False
    for line in lines:
        if line.strip().startswith("TARGET_ATTRIBUTE ="):
            new_lines.append(f'TARGET_ATTRIBUTE = "{new_attribute}"\n')
            updated = True
        else:
            new_lines.append(line)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    if updated:
        print(f"  -> Updated {file_path} to '{new_attribute}'")

def get_best_unit_from_txt(txt_path):
    """分析結果テキストからベストなLayerとUnitを抜き出す"""
    if not os.path.exists(txt_path):
        print(f"  [ERROR] Stats file not found: {txt_path}")
        return None, None

    with open(txt_path, 'r') as f:
        content = f.read()

    # DINO用のテキストフォーマットに対応
    pattern = r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)"
    match = re.search(pattern, content)
    
    if not match:
        # 他のモデル用フォーマットの可能性を考慮
        pattern = r"GLOBAL BEST SAE \(DINO\): Layer (\d+), Unit (\d+)"
        match = re.search(pattern, content)

    if match:
        layer = int(match.group(1))
        unit = int(match.group(2))
        return layer, unit
    return None, None

# --- DINO専用 ヒートマップ生成関数 ---
def generate_dino_heatmap(attribute, layer, unit_id, analysis_path):
    print(f"  -> Generating Consistent DINO Heatmap | {attribute} | L{layer} U{unit_id}...")
    
    paths_txt_path = os.path.join(analysis_path, f"top_images_paths_{attribute}.txt")
    if not os.path.exists(paths_txt_path):
        print(f"  [ERROR] Top image paths file not found: {paths_txt_path}")
        return

    with open(paths_txt_path, 'r') as f:
        target_image_paths = [line.strip() for line in f.readlines() if line.strip()]

    save_dir = os.path.join(analysis_path, "patch_heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. DINOモデルロード
    print(f"  -> Loading DINO model: {cfg_dino.MODEL_NAME}...")
    vit_model = timm.create_model(cfg_dino.MODEL_NAME, pretrained=True).to(DEVICE)
    vit_model.eval()

    # 2. SAEロード
    sae_path = cfg_dino.SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer)
    if not os.path.exists(sae_path):
        print(f"  [SKIP] SAE weights not found: {sae_path}")
        return

    sae_model = SparseAutoencoder(cfg_dino.D_MODEL, cfg_dino.D_SAE, 0.0).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_path, map_location=DEVICE))
    sae_model.eval()

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    activations = {}
    def get_act(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    for i, img_path in enumerate(target_image_paths[:9]):
        try:
            raw_img = Image.open(img_path).convert('RGB')
        except: continue
        img_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            hook = vit_model.blocks[layer].mlp.fc2.register_forward_hook(get_act("fc2"))
            vit_model(img_tensor)
            hook.remove()
            
            # SAE処理 (CLS除外)
            raw_feats = activations["fc2"][:, 1:, :] 
            B, N, D = raw_feats.shape
            _, sae_flat = sae_model(raw_feats.reshape(-1, D))
            sae_feats = sae_flat.reshape(B, N, cfg_dino.D_SAE)
            
            target_map = sae_feats[0, :, unit_id]
            score = target_map.max().item()
            hm = target_map.reshape(14, 14).cpu().numpy()
            hm = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_CUBIC)
            hm = np.maximum(hm, 0)
            if hm.max() > 0: hm /= hm.max()
            
            img_np = np.array(raw_img.resize((224, 224))) / 255.0
            ax = axes[i]
            ax.imshow(img_np); ax.imshow(hm, cmap='jet', alpha=0.5)
            ax.set_title(f"Act: {score:.2f}", fontsize=9, fontweight='bold'); ax.axis('off')

    plt.suptitle(f"DINO | {attribute} | L{layer} Unit {unit_id}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"dino_heatmap_consistent_L{layer}_U{unit_id}_{attribute}.png")
    plt.savefig(save_path); plt.close()
    print(f"  -> Saved heatmap: {save_path}")

def main():
    print("=== STARTING DINO BATCH EXPERIMENT ===")
    for attr in TARGET_ATTRIBUTES:
        print(f"\n{'='*50}\n >>> Processing Attribute: {attr}\n{'='*50}")
        update_config_file(CONFIG_PATH, attr)
        ret = subprocess.run(f"python {SCRIPT_COMPARE}", shell=True)
        if ret.returncode != 0: continue

        base_dir = cfg_dino.NORMALIZE_ANALYSIS_DIR.replace(f"_{cfg_dino.TARGET_ATTRIBUTE}", f"_{attr}")
        import glob
        pattern = os.path.join(base_dir, "**", f"global_best_{attr}_stats_full.txt")
        files = glob.glob(pattern, recursive=True)
        if not files: continue
        txt_path = max(files, key=os.path.getmtime)
        layer, unit = get_best_unit_from_txt(txt_path)
        if layer is not None:
            generate_dino_heatmap(attr, layer, unit, os.path.dirname(txt_path))

if __name__ == "__main__":
    main()