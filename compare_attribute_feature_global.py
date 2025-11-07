# compare_global_best_blond.py

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

# Hook関数 (再定義: MAE活性化取得用)
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 全てのCelebA画像から活性化を収集するためのデータセット (既存の FullCelebADatasetForViz と同じ)
class FullCelebADatasetForViz(Dataset):
    def __init__(self, img_dir, attr_path, transform):
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

# MAE/SAEの平均活性化を収集するユーティリティ関数 (既存の collect_avg_activations と同じ)
def collect_avg_activations(dataloader, layer_idx, vit_model, sae_model, target_type):
    D_MLP = D_MODEL * 4
    sum_activations = torch.zeros(D_SAE if target_type == 'SAE' else D_MLP).to(DEVICE)
    patch_count = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            activations = {}
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

            sum_activations.add_(features.sum(dim=0))
            patch_count += features.shape[0]
            
    return sum_activations / patch_count


def compare_global_best_atrtribute(num_images_to_visualize=16):
    
    # 1. パスの設定とモデル初期化
    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    vit_model.eval()
    
    dataloader_attr, dataloader_non_attr = get_celeba_attribute_loaders(
        CELEBA_IMG_DIR, CELEBA_ATTR_PATH, BATCH_SIZE, RANDOM_SEED, NUM_IMAGES_TO_SAMPLE
    )
    
    # 全層のスコアを保持するリスト
    global_scores_sae = [] # [ (score, layer_idx, feature_idx), ... ]
    global_scores_mae = [] # [ (score, layer_idx, neuron_idx), ... ]

    print("--- 1. Global Score Calculation Across All 12 Layers ---")

    for layer_idx in range(12):
        sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)
        if not os.path.exists(sae_weight_path):
            print(f"Warning: SAE weights not found for Layer {layer_idx}. Skipping.")
            continue
            
        # SAEモデルを層ごとにロード
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
        sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))
        sae_model.eval()

        # 4. SAE特徴の特定
        avg_activations_sae_attr = collect_avg_activations(dataloader_attr, layer_idx, vit_model, sae_model, 'SAE')
        avg_activations_sae_non_attr = collect_avg_activations(dataloader_non_attr, layer_idx, vit_model, sae_model, 'SAE')
        diff_score_sae = avg_activations_sae_attr - avg_activations_sae_non_attr
        
        # トップ1の特徴のスコアとインデックスを取得
        max_score_sae, max_idx_sae = torch.max(diff_score_sae, dim=0)
        global_scores_sae.append((max_score_sae.item(), layer_idx, max_idx_sae.item()))

        # 5. MAEニューロンの特定
        avg_activations_mae_attr = collect_avg_activations(dataloader_attr, layer_idx, vit_model, sae_model, 'MAE')
        avg_activations_mae_non_attr = collect_avg_activations(dataloader_non_attr, layer_idx, vit_model, sae_model, 'MAE')
        diff_score_mae = avg_activations_mae_attr - avg_activations_mae_non_attr
        
        max_score_mae, max_idx_mae = torch.max(diff_score_mae, dim=0)
        global_scores_mae.append((max_score_mae.item(), layer_idx, max_idx_mae.item()))
        
        print(f"Layer {layer_idx}: SAE Max Score={max_score_sae.item():.6f}, MAE Max Score={max_score_mae.item():.6f}")
    
    
    # 2. グローバルトップの特定
    if not global_scores_sae or not global_scores_mae:
        print("Error: No SAE models were successfully loaded or analyzed.")
        return

    # 全層を通じて最もスコアが高いものを選択
    best_sae = max(global_scores_sae, key=lambda x: x[0])
    best_mae = max(global_scores_mae, key=lambda x: x[0])

    # グローバルトップの特徴パラメータ
    best_sae_feature_layer = best_sae[1]
    best_sae_feature_idx = best_sae[2]
    mae_neuron_layer = best_mae[1]
    mae_neuron_idx = best_mae[2]
    
    print("\n" + "="*50)
    print(f"✨ GLOBAL BEST SAE FEATURE: Layer {best_sae_feature_layer}, ID {best_sae_feature_idx} (Score: {best_sae[0]:.6f})")
    print(f"✨ GLOBAL BEST MAE NEURON: Layer {mae_neuron_layer}, ID {mae_neuron_idx} (Score: {best_mae[0]:.6f})")
    print("="*50)

    
    # 3. トップ活性化画像の収集と可視化
    
    # MAEの可視化に必要なSAEモデルをロード (MAEとSAEは異なる層かもしれないため)
    sae_model_sae_viz = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model_sae_viz.load_state_dict(torch.load(SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=best_sae_feature_layer), map_location=DEVICE))
    sae_model_sae_viz.eval()
    
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
            
            activations_sae = {}
            activations_mae = {}
            
            # SAEの活性化を収集
            hook_handle_sae_fc1 = vit_model.blocks[best_sae_feature_layer].mlp.fc1.register_forward_hook(get_activation("sae_fc1", activations_sae))
            hook_handle_sae_fc2 = vit_model.blocks[best_sae_feature_layer].mlp.fc2.register_forward_hook(get_activation("sae_fc2", activations_sae))
            
            # MAEの活性化を収集
            hook_handle_mae_fc1 = vit_model.blocks[mae_neuron_layer].mlp.fc1.register_forward_hook(get_activation("mae_fc1", activations_mae))
            hook_handle_mae_fc2 = vit_model.blocks[mae_neuron_layer].mlp.fc2.register_forward_hook(get_activation("mae_fc2", activations_mae))
            
            vit_model(images)
            hook_handle_sae_fc1.remove()
            hook_handle_sae_fc2.remove()
            hook_handle_mae_fc1.remove()
            hook_handle_mae_fc2.remove()
            
            # SAE Featureの活性化 (SAEのベスト層を使用)
            sae_output = activations_sae["sae_fc2"].view(-1, D_MODEL) 
            _, sae_features = sae_model_sae_viz(sae_output)
            sae_features = sae_features.view(images.shape[0], -1, D_SAE)
            max_act_sae, _ = torch.max(sae_features[:, :, best_sae_feature_idx], dim=1)
            all_activations_sae.append(max_act_sae.cpu())
            
            # MAE Neuronの活性化 (MAEのベスト層を使用)
            mae_output = activations_mae["mae_fc1"] 
            max_act_mae, _ = torch.max(mae_output[:, :, mae_neuron_idx].abs(), dim=1)
            all_activations_mae.append(max_act_mae.cpu())
            
            all_image_paths.extend(paths)

    # トップk画像のインデックスをグローバルに特定
    all_activations_sae_tensor = torch.cat(all_activations_sae, dim=0)
    all_activations_mae_tensor = torch.cat(all_activations_mae, dim=0)
    
    total_images = all_activations_sae_tensor.size(0)
    k_viz = min(num_images_to_visualize, total_images)

    _, top_idx_sae = torch.topk(all_activations_sae_tensor, k=k_viz)
    _, top_idx_mae = torch.topk(all_activations_mae_tensor, k=k_viz)

    # 4. 比較可視化
    cols = 3
    rows_per_unit = int(ceil(k_viz / cols))
    total_rows = rows_per_unit * 2

    fig, axes = plt.subplots(total_rows, cols, figsize=(10, total_rows * 5)) # グリッドサイズを動的に設定
    axes = axes.flatten()
    print("--- 4. Generating Global Comparison Grid ---")
    
    total_plots = total_rows * cols # グリッド総枠数

    # 上半分: MAE Neuron のトップ画像
    for i in range(k_viz):
        global_idx = top_idx_mae[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')

        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[i].imshow(image_transformed)
        axes[i].set_title(f"MAE (L{mae_neuron_layer} N{mae_neuron_idx})", fontsize=8)
        axes[i].axis('off')

    # 下半分: SAE Feature のトップ画像
    for i in range(k_viz):
        global_idx = top_idx_sae[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')

        # 配置開始位置
        ax_idx = i + (rows_per_unit * cols)

        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[ax_idx].imshow(image_transformed)
        axes[ax_idx].set_title(f"SAE (L{best_sae_feature_layer} F{best_sae_feature_idx})", fontsize=8)
        axes[ax_idx].axis('off')

    # 余った軸は非表示にする
    for j in range(k_viz, rows_per_unit * cols):
        axes[j].axis('off')

    # SAEの末尾 (k_viz から rows_per_unit * cols まで)
    # SAEの開始インデックスを考慮
    for j in range(k_viz, rows_per_unit * cols):
        axes[j + (rows_per_unit * cols)].axis('off')

    fig.suptitle(f"Global Best {TARGET_ATTRIBUTE} Feature Comparison: MAE L{mae_neuron_layer} N{mae_neuron_idx} vs. SAE L{best_sae_feature_layer} F{best_sae_feature_idx}")
    plt.tight_layout()

    # 可視化の保存
    save_path = os.path.join(ANALYSIS_PATH, f"global_best_{TARGET_ATTRIBUTE}_comparison.png")
    plt.savefig(save_path)
    print(f"\nVisualization saved to {save_path}")
    plt.close('all')

    # スコア情報をテキストファイルとして保存
    score_path = os.path.join(ANALYSIS_PATH, f"global_best_{TARGET_ATTRIBUTE}_scores.txt")
    with open(score_path, 'w') as f:
        f.write("--- Global Score Calculation Across All 12 Layers ---\n")
        for score_sae, score_mae in zip(global_scores_sae, global_scores_mae):
            f.write(f"Layer {score_sae[1]}: SAE Max Score={score_sae[0]:.6f}, MAE Max Score={score_mae[0]:.6f}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write(f"GLOBAL BEST SAE FEATURE: Layer {best_sae_feature_layer}, ID {best_sae_feature_idx} (Score: {best_sae[0]:.6f})\n")
        f.write(f"GLOBAL BEST MAE NEURON: Layer {mae_neuron_layer}, ID {mae_neuron_idx} (Score: {best_mae[0]:.6f})\n")
        f.write("="*50 + "\n")
    print(f"Scores saved to {score_path}")

if __name__ == "__main__":
    compare_global_best_atrtribute(NUM_IMAGES_TO_VISUALIZE)