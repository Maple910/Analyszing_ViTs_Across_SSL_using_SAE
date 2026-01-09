import os
import re
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from sae_model import SparseAutoencoder

# ==========================================
# ★設定項目：run_all_model_ablations.py と同じパスを指定してください
# ==========================================
TARGET_ATTRIBUTE = "Person"

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

# 3. 結果の保存先
SAVE_DIR = f"./data/analysis_comparison_ablations/mae11_moco1_dino1_beit1/{TARGET_ATTRIBUTE}/visual_plots"

# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_unit_from_file(stats_path):
    if not os.path.exists(stats_path): return None, None
    with open(stats_path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.search(r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)", content)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def load_backbone(model_type):
    if model_type == "MAE":
        return timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE).eval()
    elif model_type == "DINO":
        return timm.create_model("vit_base_patch16_224.dino", pretrained=True).to(DEVICE).eval()
    elif model_type == "BEiT":
        return timm.create_model("beit_base_patch16_224", pretrained=True).to(DEVICE).eval()
    elif model_type == "MoCo":
        model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
        url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
        sd = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        nsd = {k.replace("module.base_encoder.", ""): v for k, v in sd.items() if k.startswith("module.base_encoder.")}
        model.load_state_dict(nsd, strict=False)
        return model.eval()

def get_heatmap(model, sae, img_tensor, layer, unit_id):
    """現在の状態でのヒートマップと最大活性値を返す"""
    activations = {}
    handle = model.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: activations.update({'f': o.detach()}))
    with torch.no_grad():
        model(img_tensor)
    handle.remove()
    
    _, sf_all = sae(activations['f'][:, 1:, :].reshape(-1, 768))
    target_unit_act = sf_all.view(196, -1)[:, unit_id]
    
    max_val = target_unit_act.max().item()
    max_patch_idx = torch.argmax(target_unit_act).item()
    
    # ヒートマップ用NumPy配列
    hm = target_unit_act.reshape(14, 14).detach().cpu().numpy()
    hm = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_CUBIC)
    hm = np.maximum(hm, 0)
    return hm, max_val, max_patch_idx

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tr = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 解析対象モデル
    tasks = [("MAE", MAE_PATHS), ("MoCo", MOCO_PATHS), ("BEiT", BEIT_PATHS), ("DINO", DINO_PATHS)]

    # 代表としてMAEの画像リストから1枚選んで可視化（Top-1画像）
    with open(MAE_PATHS["image_list"], 'r') as f:
        target_img_path = f.readlines()[0].strip()
    
    print(f"Analyzing visual decay for image: {os.path.basename(target_img_path)}")
    raw_img = Image.open(target_img_path).convert('RGB')
    img_t = tr(raw_img).unsqueeze(0).to(DEVICE)
    img_display = np.array(raw_img.resize((224, 224))) / 255.0

    # 描画設定: 4行(モデル) × 3列(Before, Ablation Spot, After)
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for row_idx, (name, paths) in enumerate(tasks):
        print(f">>> Processing {name}...")
        layer, unit = parse_unit_from_file(paths["stats_path"])
        if layer is None: continue
        
        # モデルとSAEのロード
        model = load_backbone(name)
        sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
        sae.load_state_dict(torch.load(os.path.join(paths["weights_dir"], f"sae_layer_{layer}.pth"), map_location=DEVICE))
        sae.eval()

        # 1. 消去前のヒートマップ取得
        hm_before, val_before, max_idx = get_heatmap(model, sae, img_t, layer, unit)
        
        # 2. 消去位置の特定と画像加工
        gy, gx = divmod(max_idx, 14)
        # 消去箇所を可視化するための画像（元の画像に赤い枠を描く）
        img_spot = raw_img.resize((224, 224)).copy()
        draw = ImageDraw.Draw(img_spot)
        # 14x14パッチなので 1パッチ = 16px
        draw.rectangle([gx*16, gy*16, (gx+1)*16, (gy+1)*16], outline="red", width=2)
        
        # 3. パッチ消去実行
        ablated_t = img_t.clone()
        ablated_t[:, :, gy*16:(gy+1)*16, gx*16:(gx+1)*16] = 0
        
        # 4. 消去後のヒートマップ取得
        hm_after, val_after, _ = get_heatmap(model, sae, ablated_t, layer, unit)
        drop_rate = (val_before - val_after) / (val_before + 1e-8) * 100

        # --- 描画 ---
        # 左列: Before
        ax_b = axes[row_idx, 0]
        ax_b.imshow(img_display)
        ax_b.imshow(hm_before / (hm_before.max() + 1e-8), cmap='jet', alpha=0.5)
        ax_b.set_title(f"{name} (Before)\nMaxAct: {val_before:.2f}")
        
        # 中列: 消去箇所（赤枠）
        ax_s = axes[row_idx, 1]
        ax_s.imshow(np.array(img_spot))
        ax_s.set_title(f"Ablated Patch Index: {max_idx}")
        
        # 右列: After
        ax_a = axes[row_idx, 2]
        ax_a.imshow(img_display)
        # Beforeのスケールに合わせて表示すると減衰がわかりやすい
        ax_a.imshow(hm_after / (hm_before.max() + 1e-8), cmap='jet', alpha=0.5)
        ax_a.set_title(f"{name} (After)\nDrop: {drop_rate:.1f}%")

        for ax in axes[row_idx]: ax.axis('off')
        
        del model, sae

    plt.suptitle(f"Visual Ablation Analysis: '{TARGET_ATTRIBUTE}'\nImpact of removing the most active patch", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(SAVE_DIR, f"visual_ablation_{TARGET_ATTRIBUTE}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[Success] 可視化画像を保存しました: {save_path}")

if __name__ == "__main__":
    main()