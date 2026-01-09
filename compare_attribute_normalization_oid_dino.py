# compare_attribute_normalization_oid_dino.py
import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import ceil
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# DINO 設定とモデル
from sae_model import SparseAutoencoder
from config_dino import *
from data_loader_oid import get_openimages_attribute_loaders

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

L1_COEFF_DUMMY = 0.0
OID_LABELS_CSV = os.path.join(OID_TRAIN_DIR, "labels.csv")
OID_IMAGES_DIR = os.path.join(OID_TRAIN_DIR, "dataset_images")

def get_activation(name, activations):
    def hook(m, i, o): activations[name] = o.detach()
    return hook

class OIDFullVizDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path); self.img_dir = img_dir; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; fn = row['filename']; p = os.path.join(self.img_dir, fn)
        try:
            img = Image.open(p).convert('RGB'); return self.transform(img) if self.transform else img, p, True
        except: return torch.zeros(3, 224, 224), p, False

def collect_stats_norm(dataloader, layer_idx, vit_model, sae_model, target_type):
    all_image_maxs = []
    with torch.no_grad():
        for images, _ in dataloader: # ★修正: 2値（images, labels）をアンパック
            images = images.to(DEVICE); activations = {}
            h_fc2 = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(get_activation("fc2", activations))
            vit_model(images); h_fc2.remove(); B, T, D = activations["fc2"].shape
            if target_type == 'SAE':
                _, features = sae_model(activations["fc2"].view(-1, D)); feat_dim = D_SAE
            else: # DINO MLP Unit
                h_fc1 = vit_model.blocks[layer_idx].mlp.fc1.register_forward_hook(get_activation("fc1", activations))
                vit_model(images); h_fc1.remove(); features = activations["fc1"].view(-1, D_MODEL*4).abs(); feat_dim = D_MODEL*4
            features = features.view(B, T, feat_dim)[:, 1:, :].max(dim=1)[0]
            all_image_maxs.append(features.cpu())
    return torch.cat(all_image_maxs, dim=0)

def compare_global_best_attribute(num_images_to_visualize=9):
    os.makedirs(NORMALIZE_PATH, exist_ok=True)
    print(f"Loading DINO model: {MODEL_NAME}")
    vit_model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE).eval()
    dl_p, dl_n = get_openimages_attribute_loaders(OID_TRAIN_DIR, TARGET_ATTRIBUTE, BATCH_SIZE, RANDOM_SEED, 2000)
    full_labels_df = pd.read_csv(OID_LABELS_CSV).set_index("filename")
    
    global_scores_sae, global_scores_dino = [], [] 
    print(f"--- 1. Global Score Calculation for DINO: {TARGET_ATTRIBUTE} ---")
    for layer_idx in range(12):
        path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)
        if not os.path.exists(path): continue
        sae = SparseAutoencoder(D_MODEL, D_SAE, 0.0).to(DEVICE)
        sae.load_state_dict(torch.load(path, map_location=DEVICE))

        for t_type, g_list in [('SAE', global_scores_sae), ('DINO', global_scores_dino)]:
            ap = collect_stats_norm(dl_p, layer_idx, vit_model, sae, t_type)
            an = collect_stats_norm(dl_n, layer_idx, vit_model, sae, t_type)
            std = torch.std(torch.cat([ap, an], dim=0), dim=0) + 1e-8
            scores = (ap.mean(0) - an.mean(0)) / std
            bv, bi = torch.max(scores, 0)
            g_list.append((bv.item(), layer_idx, bi.item(), ap[:, bi.item()].max().item()))
        print(f"Layer {layer_idx:2d} | SAE Unit {global_scores_sae[-1][2]:5d} (Score: {global_scores_sae[-1][0]:.4f})")

    best_sae, best_dino = max(global_scores_sae, key=lambda x: x[0]), max(global_scores_dino, key=lambda x: x[0])
    sae_v = SparseAutoencoder(D_MODEL, D_SAE, 0.0).to(DEVICE)
    sae_v.load_state_dict(torch.load(SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=best_sae[1]), map_location=DEVICE)); sae_v.eval()

    dataloader_viz = DataLoader(OIDFullVizDataset(OID_LABELS_CSV, OID_IMAGES_DIR, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_act_sae, all_act_dino, all_paths = [], [], []
    with torch.no_grad():
        for images, paths, valids in tqdm(dataloader_viz, desc="Scanning"):
            images = images.to(DEVICE); activations = {}
            h_s = vit_model.blocks[best_sae[1]].mlp.fc2.register_forward_hook(get_activation("s", activations))
            h_d = vit_model.blocks[best_dino[1]].mlp.fc1.register_forward_hook(get_activation("d", activations))
            vit_model(images); h_s.remove(); h_d.remove(); B = images.shape[0]
            _, sf = sae_v(activations["s"].view(-1, D_MODEL))
            s_act = sf.view(B, -1, D_SAE)[:, 1:, best_sae[2]].max(dim=1)[0].cpu()
            d_act = activations["d"].view(B, -1, D_MODEL*4).abs()[:, 1:, best_dino[2]].max(dim=1)[0].cpu()
            for i, v in enumerate(valids):
                if not v: s_act[i] = -99999.0; d_act[i] = -99999.0
            all_act_sae.append(s_act); all_act_dino.append(d_act); all_paths.extend(paths)

    top_idx_sae = torch.topk(torch.cat(all_act_sae), k=num_images_to_visualize)[1]
    top_idx_dino = torch.topk(torch.cat(all_act_dino), k=num_images_to_visualize)[1]

    cols = 3; rows = ceil(num_images_to_visualize / cols)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(10, rows * 8.0)); axes = axes.flatten()
    for i, (idx_list, layer, unit, title_p, offset, full_acts) in enumerate([
        (top_idx_dino, best_dino[1], best_dino[2], "DINO Original", 0, torch.cat(all_act_dino)), 
        (top_idx_sae, best_sae[1], best_sae[2], "SAE", rows*cols, torch.cat(all_act_sae))
    ]):
        for j in range(num_images_to_visualize):
            idx = idx_list[j].item(); p = all_paths[idx]; fn = os.path.basename(p); score = full_acts[idx].item()
            is_pos = TARGET_ATTRIBUTE in str(full_labels_df.loc[fn, 'labels']) if fn in full_labels_df.index else False
            ax = axes[offset + j]; ax.axis('off')
            try: img = Image.open(p).convert('RGB').resize((224, 224)); ax.imshow(img)
            except: ax.text(0.5, 0.5, "Error", ha='center')
            ax.set_title(f"{title_p} L{layer} U{unit}\n{'POS' if is_pos else 'NEG'} | Act: {score:.2f}", fontsize=9, color="green" if is_pos else "red", fontweight='bold')
    
    plt.tight_layout(); plt.savefig(os.path.join(NORMALIZE_PATH, f"global_best_{TARGET_ATTRIBUTE}_comparison_full.png")); plt.close()

    with open(os.path.join(NORMALIZE_PATH, f"top_images_paths_{TARGET_ATTRIBUTE}.txt"), 'w') as f:
        for i in top_idx_sae: f.write(all_paths[i.item()] + "\n")
    
    # --- 4. 統計情報の保存 (Original & SAE 両対応版) ---
    txt_path = os.path.join(NORMALIZE_PATH, f"global_best_{TARGET_ATTRIBUTE}_stats_full.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Analysis for: {TARGET_ATTRIBUTE} (DINO) ===\n\n")
        
        # 全レイヤーを通したグローバルベスト
        f.write(f"GLOBAL BEST ORIGINAL (MLP): Layer {best_dino[1]}, Unit {best_dino[2]} (Score: {best_dino[0]:.4f})\n")
        f.write(f"GLOBAL BEST SAE: Layer {best_sae[1]}, Unit {best_sae[2]} (Score: {best_sae[0]:.4f})\n\n")
        
        # 各レイヤーごとのベストユニット一覧
        f.write("--- Layer-wise Best Units (Normalized Score) ---\n")
        f.write(f"{'Layer':<6} | {'SAE Unit':<10} | {'SAE Score':<10} | {'DINO Unit':<10} | {'DINO Score':<10}\n")
        f.write("-" * 65 + "\n")
        
        for s, d in zip(global_scores_sae, global_scores_dino):
            f.write(f"Layer {s[1]:2d} | {s[2]:<10d} | {s[0]:.4f}     | {d[2]:<10d} | {d[0]:.4f}\n")

if __name__ == "__main__":
    compare_global_best_attribute(9)