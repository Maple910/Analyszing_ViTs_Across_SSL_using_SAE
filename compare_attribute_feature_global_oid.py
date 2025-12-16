# compare_attribute_feature_global_oid.py
# Open Image Dataset可視化

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

# OID用設定とモデル
from sae_model import SparseAutoencoder
from config_oid import * 
from data_loader_oid import get_openimages_attribute_loaders

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

# 平均，最大値を収集
def collect_activation_stats(dataloader, layer_idx, vit_model, sae_model, target_type):
    """
    平均活性化ベクトルと、最大活性化ベクトル(各ユニットの最大値)の両方を返す
    """
    D_MLP = D_MODEL * 4
    dim = D_SAE if target_type == 'SAE' else D_MLP
    
    sum_activations = torch.zeros(dim).to(DEVICE)
    max_activations = torch.zeros(dim).to(DEVICE) # 最大値を記録用
    patch_count = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            activations = {}
            
            # Hook登録
            hook_handle_fc1 = vit_model.blocks[layer_idx].mlp.fc1.register_forward_hook(
                get_activation(f"layer_{layer_idx}_fc1", activations)
            )
            hook_handle_fc2 = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(
                get_activation(f"layer_{layer_idx}_fc2", activations)
            )
            
            vit_model(images)
            
            hook_handle_fc1.remove()
            hook_handle_fc2.remove()
            
            layer_output = activations[f"layer_{layer_idx}_fc2"].view(-1, D_MODEL)
            
            if target_type == 'SAE':
                _, features = sae_model(layer_output)
            else: # MAE
                features = activations[f"layer_{layer_idx}_fc1"].view(-1, D_MLP)
                features = features.abs()

            # 平均用: 足し合わせる
            sum_activations.add_(features.sum(dim=0))
            patch_count += features.shape[0]
            
            # 最大値用: 現在のバッチの最大と比較して更新
            batch_max, _ = features.max(dim=0)
            max_activations = torch.max(max_activations, batch_max)
            
    return sum_activations / patch_count, max_activations

def compare_global_best_attribute(num_images_to_visualize=9):
    
    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    
    print(f"Loading ViT MAE model...")
    vit_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE)
    vit_model.eval()
    
    # スコア計算用データ (Pos/Neg 各2000枚)
    print(f"Loading Balanced Attribute Data for Scoring: {TARGET_ATTRIBUTE}")
    dataloader_attr, dataloader_non_attr = get_openimages_attribute_loaders(
        OID_BASE_DIR, TARGET_ATTRIBUTE, BATCH_SIZE, RANDOM_SEED, None
    )
    
    # 検証用データ (全量)
    if not os.path.exists(OID_LABELS_CSV):
        print(f"Error: {OID_LABELS_CSV} not found.")
        return
    full_labels_df = pd.read_csv(OID_LABELS_CSV)
    full_labels_df.set_index("filename", inplace=True)
    
    # スコア保持用リスト
    # (Score, Layer, Index, MaxActivation) を保存するように拡張
    global_scores_sae = [] 
    global_scores_mae = [] 

    print(f"--- 1. Global Score Calculation (Avg Diff & Max Act) ---")

    for layer_idx in range(12):
        sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)
        if not os.path.exists(sae_weight_path):
            continue
            
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF_DUMMY).to(DEVICE)
        sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))
        sae_model.eval()

        # SAE: 平均と最大値を取得
        avg_sae_pos, max_sae_pos = collect_activation_stats(dataloader_attr, layer_idx, vit_model, sae_model, 'SAE')
        avg_sae_neg, _           = collect_activation_stats(dataloader_non_attr, layer_idx, vit_model, sae_model, 'SAE')
        
        diff_score_sae = avg_sae_pos - avg_sae_neg
        
        # 最も差分が大きいユニットを選択
        best_score_sae, best_idx_sae = torch.max(diff_score_sae, dim=0)
        best_idx_sae = best_idx_sae.item()
        best_score_sae = best_score_sae.item()
        
        # そのユニットのPositiveデータでの最大活性化値を取得
        best_max_act_sae = max_sae_pos[best_idx_sae].item()
        
        global_scores_sae.append((best_score_sae, layer_idx, best_idx_sae, best_max_act_sae))

        # MAE: 平均と最大値を取得
        avg_mae_pos, max_mae_pos = collect_activation_stats(dataloader_attr, layer_idx, vit_model, sae_model, 'MAE')
        avg_mae_neg, _           = collect_activation_stats(dataloader_non_attr, layer_idx, vit_model, sae_model, 'MAE')
        
        diff_score_mae = avg_mae_pos - avg_mae_neg
        
        best_score_mae, best_idx_mae = torch.max(diff_score_mae, dim=0)
        best_idx_mae = best_idx_mae.item()
        best_score_mae = best_score_mae.item()
        
        best_max_act_mae = max_mae_pos[best_idx_mae].item()
        
        global_scores_mae.append((best_score_mae, layer_idx, best_idx_mae, best_max_act_mae))
        
        print(f"Layer {layer_idx}: "
              f"SAE Unit {best_idx_sae} (Score: {best_score_sae:.6f}, MaxAct: {best_max_act_sae:.6f}), "
              f"MAE Unit {best_idx_mae} (Score: {best_score_mae:.6f}, MaxAct: {best_max_act_mae:.6f})")

    if not global_scores_sae:
        return

    # グローバルトップの特定 (スコア基準)
    best_sae = max(global_scores_sae, key=lambda x: x[0])
    best_mae = max(global_scores_mae, key=lambda x: x[0])

    # (Score, Layer, Index, MaxAct)
    best_sae_layer, best_sae_idx = best_sae[1], best_sae[2]
    best_mae_layer, best_mae_idx = best_mae[1], best_mae[2]
    
    print("\n" + "="*50)
    print(f"✨ GLOBAL BEST SAE: Layer {best_sae_layer}, Unit {best_sae_idx}")
    print(f"   Score: {best_sae[0]:.6f}, Max Act (Pos): {best_sae[3]:.6f}")
    print(f"✨ GLOBAL BEST MAE: Layer {best_mae_layer}, Unit {best_mae_idx}")
    print(f"   Score: {best_mae[0]:.6f}, Max Act (Pos): {best_mae[3]:.6f}")
    print("="*50)

    # 4. 可視化データの収集 (全量スキャン)
    viz_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset_viz = OIDFullVizDataset(OID_LABELS_CSV, OID_IMAGES_DIR, transform=viz_transform)
    dataloader_viz = DataLoader(dataset_viz, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 可視化用SAEロード
    sae_model_viz = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF_DUMMY).to(DEVICE)
    sae_model_viz.load_state_dict(torch.load(SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=best_sae_layer), map_location=DEVICE))
    sae_model_viz.eval()

    all_activations_sae = []
    all_activations_mae = []
    all_image_paths = []

    print("--- 3. Collecting Top Activation Images (Scanning Full Training Set) ---")
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader_viz, desc="Scanning 500k Images"):
            images = images.to(DEVICE)
            activations = {}
            
            h_sae_fc1 = vit_model.blocks[best_sae_layer].mlp.fc1.register_forward_hook(get_activation("sae_fc1", activations))
            h_sae_fc2 = vit_model.blocks[best_sae_layer].mlp.fc2.register_forward_hook(get_activation("sae_fc2", activations))
            h_mae_fc1 = vit_model.blocks[best_mae_layer].mlp.fc1.register_forward_hook(get_activation("mae_fc1", activations))
            
            vit_model(images)
            h_sae_fc1.remove(); h_sae_fc2.remove(); h_mae_fc1.remove()
            
            # SAE
            sae_in = activations["sae_fc2"].view(-1, D_MODEL)
            _, sae_feats = sae_model_viz(sae_in)
            sae_feats = sae_feats.view(images.shape[0], -1, D_SAE)
            max_sae, _ = torch.max(sae_feats[:, :, best_sae_idx], dim=1)
            all_activations_sae.append(max_sae.cpu())
            
            # MAE
            mae_feats = activations["mae_fc1"].view(images.shape[0], -1, D_MODEL*4).abs()
            max_mae, _ = torch.max(mae_feats[:, :, best_mae_idx], dim=1)
            all_activations_mae.append(max_mae.cpu())
            
            all_image_paths.extend(paths)

    all_activations_sae = torch.cat(all_activations_sae)
    all_activations_mae = torch.cat(all_activations_mae)
    
    k_viz = min(num_images_to_visualize, len(all_image_paths))
    _, top_idx_sae = torch.topk(all_activations_sae, k=k_viz)
    _, top_idx_mae = torch.topk(all_activations_mae, k=k_viz)

    # Precision Check
    def check_target_class(indices, all_paths, df):
        match_count = 0
        total = len(indices)
        for i in indices:
            path = all_paths[i.item()]
            filename = os.path.basename(path)
            try:
                if TARGET_ATTRIBUTE in str(df.loc[filename, 'labels']):
                    match_count += 1
            except KeyError: pass 
        return match_count, total

    sae_match, sae_total = check_target_class(top_idx_sae, all_image_paths, full_labels_df)
    mae_match, mae_total = check_target_class(top_idx_mae, all_image_paths, full_labels_df)
    
    sae_precision = (sae_match / sae_total) * 100 if sae_total > 0 else 0
    mae_precision = (mae_match / mae_total) * 100 if mae_total > 0 else 0

    msg_sae = f"SAE Top-{k_viz} Selectivity: {sae_precision:.1f}% ({sae_match}/{sae_total})"
    msg_mae = f"MAE Top-{k_viz} Selectivity: {mae_precision:.1f}% ({mae_match}/{mae_total})"
    
    print(f"\n[Verification] {msg_sae}")
    print(f"[Verification] {msg_mae}")

    # 5. 可視化生成
    cols = 3
    rows = ceil(k_viz / cols)
    total_rows = rows * 2
    
    fig, axes = plt.subplots(total_rows, cols, figsize=(10, total_rows * 4.0))
    axes = axes.flatten()
    
    # MAE Images
    for i in range(k_viz):
        idx = top_idx_mae[i].item()
        path = all_image_paths[idx]
        score = all_activations_mae[idx].item()
        filename = os.path.basename(path)
        
        is_pos = False
        try:
            if TARGET_ATTRIBUTE in str(full_labels_df.loc[filename, 'labels']): is_pos = True
        except: pass
        
        color = "green" if is_pos else "red"
        img = Image.open(path).convert('RGB')
        img = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(img)
        axes[i].imshow(img)
        axes[i].set_title(f"MAE L{best_mae_layer} N{best_mae_idx}\n{'POS' if is_pos else 'NEG'} | Act: {score:.2f}", 
                          fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')
        
    # SAE Images
    start_sae = rows * cols
    for i in range(k_viz):
        idx = top_idx_sae[i].item()
        path = all_image_paths[idx]
        score = all_activations_sae[idx].item()
        filename = os.path.basename(path)
        
        is_pos = False
        try:
            if TARGET_ATTRIBUTE in str(full_labels_df.loc[filename, 'labels']): is_pos = True
        except: pass
        
        color = "green" if is_pos else "red"
        img = Image.open(path).convert('RGB')
        img = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(img)
        ax_idx = start_sae + i
        axes[ax_idx].imshow(img)
        axes[ax_idx].set_title(f"SAE L{best_sae_layer} F{best_sae_idx}\n{'POS' if is_pos else 'NEG'} | Act: {score:.2f}", 
                               fontsize=9, color=color, fontweight='bold')
        axes[ax_idx].axis('off')

    for i in range(len(axes)):
        if (i >= k_viz and i < start_sae) or (i >= start_sae + k_viz): axes[i].axis('off')

    plt.suptitle(f"Global Best: '{TARGET_ATTRIBUTE}'", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, f"global_best_{TARGET_ATTRIBUTE}_comparison_full.png"))
    plt.close()

    # 6. テキストファイル保存 (Max Actを追加)
    txt_path = os.path.join(ANALYSIS_PATH, f"global_best_{TARGET_ATTRIBUTE}_stats_full.txt")
    with open(txt_path, 'w') as f:
        f.write(f"=== Analysis for: {TARGET_ATTRIBUTE} ===\n\n")
        f.write(f"{msg_sae}\n{msg_mae}\n\n")

        f.write("--- Layer-wise Best Units (Score & MaxAct) ---\n")
        f.write("Note: Score = Avg(Pos) - Avg(Neg), MaxAct = Max(Pos)\n")
        for s, m in zip(global_scores_sae, global_scores_mae):
            # s = (score, layer, idx, max_act)
            f.write(f"Layer {s[1]:2d}: ")
            f.write(f"SAE Unit {s[2]:5d} (Score: {s[0]:.4f}, MaxAct: {s[3]:.4f}) | ")
            f.write(f"MAE Unit {m[2]:5d} (Score: {m[0]:.4f}, MaxAct: {m[3]:.4f})\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write(f"GLOBAL BEST SAE: Layer {best_sae_layer}, Unit {best_sae_idx}\n")
        f.write(f"  -> Score: {best_sae[0]:.6f}, Max Act (Pos): {best_sae[3]:.6f}\n")
        f.write(f"GLOBAL BEST MAE: Layer {best_mae_layer}, Unit {best_mae_idx}\n")
        f.write(f"  -> Score: {best_mae[0]:.6f}, Max Act (Pos): {best_mae[3]:.6f}\n")
        f.write("="*50 + "\n")
        
    print(f"Stats saved to: {txt_path}")

if __name__ == "__main__":
    compare_global_best_attribute(9)