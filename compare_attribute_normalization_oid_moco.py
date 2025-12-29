# compare_global_attribute_best_moco.py
# Open Image Dataset可視化 (正規化スコア & 色分け可視化)

import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import ceil
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# SAEモデル定義と設定 (MoCo用)
from sae_model import SparseAutoencoder
from config_moco import * 
from data_loader_oid import get_openimages_attribute_loaders

import torch
import numpy as np
import random
import time

# 可視化には一切使わないがモデル引数に必要なのでダミー
L1_COEFF_DUMMY = 0.0

# --- 設定 ---
OID_FULL_DATA_DIR = OID_TRAIN_DIR 
OID_LABELS_CSV = os.path.join(OID_FULL_DATA_DIR, "labels.csv")
OID_IMAGES_DIR = os.path.join(OID_FULL_DATA_DIR, "dataset_images")

# --- Hook関数 ---
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# --- 可視化用データセットクラス ---
class OIDFullVizDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        print(f"Loading metadata from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), img_path

# 正規化スコア算出のために画像単位の最大活性値を収集
def collect_activation_stats_norm(dataloader, layer_idx, vit_model, sae_model, target_type):
    D_MLP = D_MODEL * 4
    all_image_maxs = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            activations = {}
            h_fc2 = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(get_activation("fc2", activations))
            vit_model(images)
            h_fc2.remove()
            
            B, T, D = activations["fc2"].shape
            
            if target_type == 'SAE':
                _, features = sae_model(activations["fc2"].view(-1, D))
                feat_dim = D_SAE
            else: # MoCo
                h_fc1 = vit_model.blocks[layer_idx].mlp.fc1.register_forward_hook(get_activation("fc1", activations))
                vit_model(images)
                h_fc1.remove()
                features = activations["fc1"].view(-1, D_MLP).abs()
                feat_dim = D_MLP

            # (B, 197, Dim) にリシェイプし、CLSトークン(0番目)を除外して空間最大を取る
            features = features.view(B, T, feat_dim)
            img_max = features[:, 1:, :].max(dim=1)[0]
            all_image_maxs.append(img_max.cpu())
            
    return torch.cat(all_image_maxs, dim=0)

def compare_global_best_attribute(num_images_to_visualize=9):
    os.makedirs(NORMALIZE_PATH, exist_ok=True)
    
    # 1. MoCo v3 モデルのロード
    print(f"Loading MoCo v3 ViT model...")
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
    url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    new_state_dict = {k.replace("module.base_encoder.", ""): v for k, v in state_dict.items() if k.startswith("module.base_encoder.")}
    vit_model.load_state_dict(new_state_dict, strict=False)
    vit_model.eval()
    
    print(f"Loading Balanced Attribute Data for Scoring: {TARGET_ATTRIBUTE}")
    dataloader_attr, dataloader_non_attr = get_openimages_attribute_loaders(
        OID_TRAIN_DIR, TARGET_ATTRIBUTE, BATCH_SIZE, RANDOM_SEED, 2000
    )
    
    if not os.path.exists(OID_LABELS_CSV):
        print(f"Error: {OID_LABELS_CSV} not found.")
        return
    full_labels_df = pd.read_csv(OID_LABELS_CSV).set_index("filename")
    
    global_scores_sae = [] 
    global_scores_moco = [] 

    print(f"--- 1. Global Score Calculation (Normalized Diff) ---")
    for layer_idx in range(12):
        sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)
        if not os.path.exists(sae_weight_path): continue
        
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF_DUMMY).to(DEVICE).eval()
        sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))

        # SAE: 正規化スコア
        acts_pos = collect_activation_stats_norm(dataloader_attr, layer_idx, vit_model, sae_model, 'SAE')
        acts_neg = collect_activation_stats_norm(dataloader_non_attr, layer_idx, vit_model, sae_model, 'SAE')
        std_all = torch.std(torch.cat([acts_pos, acts_neg], dim=0), dim=0) + 1e-8
        norm_scores_sae = (acts_pos.mean(dim=0) - acts_neg.mean(dim=0)) / std_all
        best_val_sae, best_idx_sae = torch.max(norm_scores_sae, dim=0)
        global_scores_sae.append((best_val_sae.item(), layer_idx, best_idx_sae.item(), acts_pos[:, best_idx_sae.item()].max().item()))

        # MoCo: 正規化スコア
        m_pos = collect_activation_stats_norm(dataloader_attr, layer_idx, vit_model, sae_model, 'MoCo')
        m_neg = collect_activation_stats_norm(dataloader_non_attr, layer_idx, vit_model, sae_model, 'MoCo')
        std_m = torch.std(torch.cat([m_pos, m_neg], dim=0), dim=0) + 1e-8
        norm_scores_moco = (m_pos.mean(dim=0) - m_neg.mean(dim=0)) / std_m
        best_val_moco, best_idx_moco = torch.max(norm_scores_moco, dim=0)
        global_scores_moco.append((best_val_moco.item(), layer_idx, best_idx_moco.item(), m_pos[:, best_idx_moco.item()].max().item()))
        
        print(f"Layer {layer_idx:2d} | SAE Unit {best_idx_sae.item():5d} (NormScore: {best_val_sae.item():.4f}) | MoCo Unit {best_idx_moco.item():5d} (NormScore: {best_val_moco.item():.4f})")

    best_sae = max(global_scores_sae, key=lambda x: x[0])
    best_moco = max(global_scores_moco, key=lambda x: x[0])
    best_sae_layer, best_sae_idx = best_sae[1], best_sae[2]
    best_moco_layer, best_moco_idx = best_moco[1], best_moco[2]
    
    # 4. 可視化データの収集 (全量スキャン)
    viz_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_viz = OIDFullVizDataset(OID_LABELS_CSV, OID_IMAGES_DIR, transform=viz_transform)
    dataloader_viz = DataLoader(dataset_viz, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    sae_model_viz = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF_DUMMY).to(DEVICE)
    sae_model_viz.load_state_dict(torch.load(SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=best_sae_layer), map_location=DEVICE))
    sae_model_viz.eval()

    all_act_sae, all_act_moco, all_image_paths = [], [], []
    with torch.no_grad():
        for images, paths in tqdm(dataloader_viz, desc="Scanning for Top Images"):
            images = images.to(DEVICE); activations = {}
            h_s = vit_model.blocks[best_sae_layer].mlp.fc2.register_forward_hook(get_activation("s", activations))
            h_m = vit_model.blocks[best_moco_layer].mlp.fc1.register_forward_hook(get_activation("m", activations))
            vit_model(images); h_s.remove(); h_m.remove()
            
            B = images.shape[0]
            _, sf = sae_model_viz(activations["s"].view(-1, D_MODEL))
            sf = sf.view(B, -1, D_SAE)
            all_act_sae.append(sf[:, 1:, best_sae_idx].max(dim=1)[0].cpu())
            all_act_moco.append(activations["m"].view(B, -1, D_MODEL*4).abs()[:, 1:, best_moco_idx].max(dim=1)[0].cpu())
            all_image_paths.extend(paths)

    all_act_sae, all_act_moco = torch.cat(all_act_sae), torch.cat(all_act_moco)
    k_viz = min(num_images_to_visualize, len(all_image_paths))
    _, top_idx_sae = torch.topk(all_act_sae, k=k_viz)
    _, top_idx_moco = torch.topk(all_act_moco, k=k_viz)

    # 5. 可視化生成 (色分けロジック)
    cols = 3
    rows = ceil(k_viz / cols)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(10, rows * 8.0))
    axes = axes.flatten()
    
    for i, (idx_list, layer, unit, title_p, offset) in enumerate([
        (top_idx_moco, best_moco_layer, best_moco_idx, "MoCo", 0),
        (top_idx_sae, best_sae_layer, best_sae_idx, "SAE", rows * cols)
    ]):
        for j in range(k_viz):
            idx = idx_list[j].item()
            path = all_image_paths[idx]
            filename = os.path.basename(path)
            score = all_act_moco[idx].item() if title_p == "MoCo" else all_act_sae[idx].item()
            
            is_pos = False
            try:
                if TARGET_ATTRIBUTE in str(full_labels_df.loc[filename, 'labels']): is_pos = True
            except: pass
            
            color = "green" if is_pos else "red"
            img = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(Image.open(path).convert('RGB'))
            ax = axes[offset + j]
            ax.imshow(img); ax.axis('off')
            ax.set_title(f"{title_p} L{layer} U{unit}\n{'POS' if is_pos else 'NEG'} | Act: {score:.2f}", 
                         fontsize=9, color=color, fontweight='bold')

    plt.suptitle(f"Global Best (Normalized): '{TARGET_ATTRIBUTE}' (MoCo)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(NORMALIZE_PATH, f"global_best_{TARGET_ATTRIBUTE}_comparison_full.png"))
    plt.close()

    # 統計保存 (形式維持)
    txt_path = os.path.join(NORMALIZE_PATH, f"global_best_{TARGET_ATTRIBUTE}_stats_full.txt")
    with open(txt_path, 'w') as f:
        f.write(f"=== Analysis for: {TARGET_ATTRIBUTE} (MoCo) ===\n\n")
        f.write("--- Layer-wise Best Units (Normalized Score) ---\n")
        for s, m in zip(global_scores_sae, global_scores_moco):
            f.write(f"Layer {s[1]:2d}: SAE Unit {s[2]:5d} (Score: {s[0]:.4f}, Max: {s[3]:.4f}) | MoCo Unit {m[2]:5d} (Score: {m[0]:.4f}, Max: {m[3]:.4f})\n")
        f.write("\n" + "="*50 + "\n")
        f.write(f"GLOBAL BEST SAE: Layer {best_sae_layer}, Unit {best_sae_idx}\n")
        f.write(f"  -> Score: {best_sae[0]:.6f}, Max Act (Pos): {best_sae[3]:.6f}\n")
        f.write(f"GLOBAL BEST MoCo: Layer {best_moco_layer}, Unit {best_moco_idx}\n")
        f.write(f"  -> Score: {best_moco[0]:.6f}, Max Act (Pos): {best_moco[3]:.6f}\n")
        f.write("="*50 + "\n")

    print(f"Stats saved to: {txt_path}")
    
    # ★最重要: ヒートマップ作成用に画像パスを保存
    paths_txt_path = os.path.join(NORMALIZE_PATH, f"top_images_paths_{TARGET_ATTRIBUTE}.txt")
    with open(paths_txt_path, 'w') as f:
        for i in top_idx_sae:
            f.write(all_image_paths[i.item()] + "\n")
    print(f"Top image paths saved to: {paths_txt_path}")

if __name__ == "__main__":
    compare_global_best_attribute(9)