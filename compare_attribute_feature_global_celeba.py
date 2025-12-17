# compare_attribute_feature_global_celeba.py
# CelebA可視化

import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from tqdm import tqdm
from sae_model import SparseAutoencoder
from data_loader_celeba import get_celeba_attribute_loaders
from config_celeba import * 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
from math import ceil

def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 全てのCelebA画像から活性化を収集するためのデータセット
class FullCelebADatasetForViz(Dataset):
    def __init__(self, img_dir, attr_path, transform):
        # 属性ファイルを読み込み
        df = pd.read_csv(attr_path, sep='\s+', skiprows=1)
        self.image_paths = [os.path.join(img_dir, filename) for filename in df.index]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        return image_tensor, img_path 

#平均,最大値収集
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


def compare_global_best_attribute(num_images_to_visualize=16):
    
    # 1. パスの設定とモデル初期化
    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    vit_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE)
    vit_model.eval()
    
    # 正答率チェック用の属性データ読み込み
    print(f"Loading Attribute Data for Verification: {TARGET_ATTRIBUTE}")
    attr_df_ref = pd.read_csv(CELEBA_ATTR_PATH, sep='\s+', skiprows=1, index_col=0)

    # スコア計算用データローダー (Pos/Neg)
    dataloader_attr, dataloader_non_attr = get_celeba_attribute_loaders(
        CELEBA_IMG_DIR, CELEBA_ATTR_PATH, BATCH_SIZE, RANDOM_SEED, NUM_IMAGES_TO_SAMPLE
    )
    
    # 全層のスコアを保持するリスト
    # (Score, Layer, Index, MaxActivation) を保存
    global_scores_sae = [] 
    global_scores_mae = [] 

    print("--- 1. Global Score Calculation (Avg Diff & Max Act) ---")

    for layer_idx in range(12):
        sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)
        if not os.path.exists(sae_weight_path):
            print(f"Warning: SAE weights not found for Layer {layer_idx}. Skipping.")
            continue
            
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
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
        print("Error: No SAE models were successfully loaded or analyzed.")
        return

    # グローバルトップの特定
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

    
    # 3. トップ活性化画像の収集と可視化
    
    sae_model_viz = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model_viz.load_state_dict(torch.load(SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=best_sae_layer), map_location=DEVICE))
    sae_model_viz.eval()
    
    # 可視化用のデータローダー (全画像)
    transform_viz = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_full = FullCelebADatasetForViz(CELEBA_IMG_DIR, CELEBA_ATTR_PATH, transform_viz)
    dataloader_full = DataLoader(dataset_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_activations_sae = []
    all_activations_mae = []
    all_image_paths = []

    print("--- 3. Collecting Top Activation Images for Global Best Features ---")
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader_full, desc="Global Best Max Act"):
            images = images.to(DEVICE)
            activations = {}
            
            # SAEの活性化を収集
            h_sae_fc1 = vit_model.blocks[best_sae_layer].mlp.fc1.register_forward_hook(get_activation("sae_fc1", activations))
            h_sae_fc2 = vit_model.blocks[best_sae_layer].mlp.fc2.register_forward_hook(get_activation("sae_fc2", activations))
            
            # MAEの活性化を収集
            h_mae_fc1 = vit_model.blocks[best_mae_layer].mlp.fc1.register_forward_hook(get_activation("mae_fc1", activations))
            
            vit_model(images)
            h_sae_fc1.remove(); h_sae_fc2.remove(); h_mae_fc1.remove()
            
            # SAE Feature
            sae_output = activations["sae_fc2"].view(-1, D_MODEL) 
            _, sae_features = sae_model_viz(sae_output)
            sae_features = sae_features.view(images.shape[0], -1, D_SAE)
            max_act_sae, _ = torch.max(sae_features[:, :, best_sae_idx], dim=1)
            all_activations_sae.append(max_act_sae.cpu())
            
            # MAE Neuron
            mae_output = activations["mae_fc1"] 
            max_act_mae, _ = torch.max(mae_output[:, :, best_mae_idx].abs(), dim=1)
            all_activations_mae.append(max_act_mae.cpu())
            
            all_image_paths.extend(paths)

    all_activations_sae_tensor = torch.cat(all_activations_sae, dim=0)
    all_activations_mae_tensor = torch.cat(all_activations_mae, dim=0)
    
    total_images = all_activations_sae_tensor.size(0)
    k_viz = min(num_images_to_visualize, total_images)

    _, top_idx_sae = torch.topk(all_activations_sae_tensor, k=k_viz)
    _, top_idx_mae = torch.topk(all_activations_mae_tensor, k=k_viz)

    # --- 正答率（Selectivity）の計算 ---
    def calculate_precision(indices, paths, attr_df, target):
        hit = 0
        total = len(indices)
        for idx in indices:
            path = paths[idx.item()]
            filename = os.path.basename(path)
            try:
                if attr_df.loc[filename, target] == 1:
                    hit += 1
            except KeyError: pass
        return hit, total

    sae_hits, sae_total = calculate_precision(top_idx_sae, all_image_paths, attr_df_ref, TARGET_ATTRIBUTE)
    mae_hits, mae_total = calculate_precision(top_idx_mae, all_image_paths, attr_df_ref, TARGET_ATTRIBUTE)

    msg_sae = f"SAE Top-{k_viz} Selectivity: {sae_hits/sae_total*100:.1f}% ({sae_hits}/{sae_total})"
    msg_mae = f"MAE Top-{k_viz} Selectivity: {mae_hits/mae_total*100:.1f}% ({mae_hits}/{mae_total})"

    print(f"\n[Verification] {msg_sae}")
    print(f"[Verification] {msg_mae}")

    # 4. 比較可視化
    cols = 3
    rows_per_unit = int(ceil(k_viz / cols))
    total_rows = rows_per_unit * 2

    fig, axes = plt.subplots(total_rows, cols, figsize=(10, total_rows * 5))
    axes = axes.flatten()
    print("--- 4. Generating Global Comparison Grid ---")
    
    # MAE Images
    for i in range(k_viz):
        global_idx = top_idx_mae[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        score = all_activations_mae_tensor[global_idx].item()
        
        filename = os.path.basename(img_path)
        is_pos = (attr_df_ref.loc[filename, TARGET_ATTRIBUTE] == 1)
        color = "green" if is_pos else "red"

        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[i].imshow(image_transformed)
        axes[i].set_title(f"MAE L{best_mae_layer} N{best_mae_idx}\n{'POS' if is_pos else 'NEG'} | Act: {score:.2f}", 
                          fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')

    # SAE Images
    start_sae = rows_per_unit * cols
    for i in range(k_viz):
        global_idx = top_idx_sae[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        score = all_activations_sae_tensor[global_idx].item()

        filename = os.path.basename(img_path)
        is_pos = (attr_df_ref.loc[filename, TARGET_ATTRIBUTE] == 1)
        color = "green" if is_pos else "red"

        ax_idx = start_sae + i
        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[ax_idx].imshow(image_transformed)
        axes[ax_idx].set_title(f"SAE L{best_sae_layer} F{best_sae_idx}\n{'POS' if is_pos else 'NEG'} | Act: {score:.2f}", 
                               fontsize=9, color=color, fontweight='bold')
        axes[ax_idx].axis('off')

    for i in range(len(axes)):
        if (i >= k_viz and i < start_sae) or (i >= start_sae + k_viz):
            axes[i].axis('off')

    fig.suptitle(f"Global Best {TARGET_ATTRIBUTE} Feature Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_PATH, f"global_best_{TARGET_ATTRIBUTE}_comparison.png"))
    plt.close()

    # スコア情報をテキストファイルとして保存 (Max Actを追加)
    score_path = os.path.join(ANALYSIS_PATH, f"global_best_{TARGET_ATTRIBUTE}_scores.txt")
    with open(score_path, 'w') as f:
        f.write(f"=== Global Best Analysis for Attribute: {TARGET_ATTRIBUTE} ===\n\n")
        
        f.write("--- Selectivity Check ---\n")
        f.write(f"Dataset Size: {len(all_image_paths)} images\n")
        f.write(f"{msg_sae}\n")
        f.write(f"{msg_mae}\n\n")

        f.write("--- Layer-wise Best Units (Score & MaxAct) ---\n")
        f.write("Note: Score = Avg(Pos) - Avg(Neg), MaxAct = Max(Pos)\n")
        for s, m in zip(global_scores_sae, global_scores_mae):
             # s = (score, layer, idx, max_act)
            f.write(f"Layer {s[1]:2d}: ")
            f.write(f"SAE Unit {s[2]:5d} (Score: {s[0]:.6f}, MaxAct: {s[3]:.6f}) | ")
            f.write(f"MAE Unit {m[2]:5d} (Score: {m[0]:.6f}, MaxAct: {m[3]:.6f})\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write(f"GLOBAL BEST SAE: Layer {best_sae_layer}, Unit {best_sae_idx}\n")
        f.write(f"  -> Score: {best_sae[0]:.6f}, Max Act (Pos): {best_sae[3]:.6f}\n")
        f.write(f"GLOBAL BEST MAE: Layer {best_mae_layer}, Unit {best_mae_idx}\n")
        f.write(f"  -> Score: {best_mae[0]:.6f}, Max Act (Pos): {best_mae[3]:.6f}\n")
        f.write("="*50 + "\n")
        
    print(f"Scores saved to {score_path}")

if __name__ == "__main__":
    compare_global_best_attribute(NUM_IMAGES_TO_VISUALIZE)