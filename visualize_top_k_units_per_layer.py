# config_celebaで指定したLAYER_TO_ANALYZE，K_TOP_UNITSを使って可視化

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
from math import ceil


def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 全てのCelebA画像から活性化を収集するためのデータセット
class FullCelebADatasetForViz(Dataset):
    def __init__(self, img_dir, attr_path, transform):
        df = pd.read_csv(attr_path, delim_whitespace=True, skiprows=1)
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

# MAE/SAEの平均活性化を収集するユーティリティ関数
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
                features = activations[f"layer_{layer_idx}_fc1"].view(-1, D_MODEL * 4)
                features = features.abs()

            sum_activations.add_(features.sum(dim=0))
            patch_count += features.shape[0]
            
    return sum_activations / patch_count

def visualize_single_unit(unit_type, layer_idx, unit_idx, rank, vit_model, sae_model, dataloader_full, num_images_to_visualize, save_dir):
    
    # 1. 保存パスの決定とスキップチェック
    # save_dir の下に unit_type 名（'MAE' または 'SAE'）のサブディレクトリを作成して保存する
    type_dir = os.path.join(save_dir, unit_type)
    os.makedirs(type_dir, exist_ok=True)

    file_name = f"L{layer_idx}_{unit_type}_Rank{rank:03d}_U{unit_idx}.png"
    save_path = os.path.join(type_dir, file_name)

    if os.path.exists(save_path):
        print(f"    -> Skipping {unit_type} Unit {unit_idx} (Rank {rank:03d}): File already exists at {save_path}.")
        return
        
    all_activations = []
    all_image_paths = []
    
    print(f"    -> Collecting global activations for {unit_type} Unit {unit_idx} (Rank {rank})...")
    
    target_layer_idx = layer_idx
    target_unit_idx = unit_idx
    
    abs_func = torch.abs if unit_type == 'MAE' else lambda x: x
    
    with torch.no_grad():
        for images, paths in dataloader_full:
            images = images.to(DEVICE)
            activations = {}
            
            hook_handle_fc1 = vit_model.blocks[target_layer_idx].mlp.fc1.register_forward_hook(
                get_activation("fc1", activations)
            )
            hook_handle_fc2 = vit_model.blocks[target_layer_idx].mlp.fc2.register_forward_hook(
                get_activation("fc2", activations)
            )
            vit_model(images)
            hook_handle_fc1.remove()
            hook_handle_fc2.remove()
            
            if unit_type == 'SAE':
                layer_output = activations["fc2"].view(-1, D_MODEL) 
                _, sae_features = sae_model(layer_output)
                sae_features = sae_features.view(images.shape[0], -1, D_SAE)
                unit_output = sae_features[:, :, target_unit_idx] 
            else: # MAE
                unit_output = activations["fc1"][:, :, target_unit_idx] 
                
            max_act, _ = torch.max(abs_func(unit_output), dim=1)
            all_activations.append(max_act.cpu())
            all_image_paths.extend(paths)

    all_activations_tensor = torch.cat(all_activations, dim=0)
    total_images = all_activations_tensor.size(0)
    k_viz = min(num_images_to_visualize, total_images)

    top_act, top_idx = torch.topk(all_activations_tensor, k=k_viz)

    # 4. 可視化
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    
    # タイトルとファイル名の設定
    unit_name = f"{unit_type} Unit (L{target_layer_idx} U{target_unit_idx}, Rank {rank})"
    
    print(f"    -> Visualizing top {k_viz} images for {unit_name}...")

    for i in range(k_viz):
        global_idx = top_idx[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        
        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[i].imshow(image_transformed)
        axes[i].set_title(f"Act: {top_act[i].item():.4f}", fontsize=8)
        axes[i].axis('off')

    # 余った軸を非表示
    for j in range(k_viz, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Top Images for {unit_name} (Attribute: {TARGET_ATTRIBUTE})")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close('all')
    print(f"    -> Visualization saved to {save_path}")


def analyze_and_visualize_top_k(layer_idx, k_top_units, num_images_to_visualize):
    
    # 1. パスの設定とモデルロード
    os.makedirs(ANALYSIS_PATH, exist_ok=True) 
    save_dir = ANALYSIS_PATH # 保存ディレクトリを設定

    sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)

    if not os.path.exists(sae_weight_path):
        print(f"Error: SAE weights not found at {sae_weight_path}. Skipping layer {layer_idx}.")
        return

    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))
    vit_model.eval()
    sae_model.eval()

    # 2. データローダーのロード
    dataloader_attr, dataloader_non_attr = get_celeba_attribute_loaders(
        CELEBA_IMG_DIR, CELEBA_ATTR_PATH, BATCH_SIZE, RANDOM_SEED, NUM_IMAGES_TO_SAMPLE
    )
    transform_viz = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_full = FullCelebADatasetForViz(CELEBA_IMG_DIR, CELEBA_ATTR_PATH, transform_viz)
    dataloader_full = DataLoader(dataset_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    
    # 3. 属性特異性スコアの計算
    print(f"--- 3. Calculating Attribute Specificity Score for Layer {layer_idx} ---")
    
    # SAE
    avg_activations_sae_attr = collect_avg_activations(dataloader_attr, layer_idx, vit_model, sae_model, 'SAE')
    avg_activations_sae_non_attr = collect_avg_activations(dataloader_non_attr, layer_idx, vit_model, sae_model, 'SAE')
    diff_score_sae = avg_activations_sae_attr - avg_activations_sae_non_attr
    top_sae_scores, top_sae_indices = torch.topk(diff_score_sae, k=k_top_units, dim=0)

    # MAE
    avg_activations_mae_attr = collect_avg_activations(dataloader_attr, layer_idx, vit_model, sae_model, 'MAE')
    avg_activations_mae_non_attr = collect_avg_activations(dataloader_non_attr, layer_idx, vit_model, sae_model, 'MAE')
    diff_score_mae = avg_activations_mae_attr - avg_activations_mae_non_attr
    top_mae_scores, top_mae_indices = torch.topk(diff_score_mae, k=k_top_units, dim=0)
    
    print(f"Layer {layer_idx} Analysis Complete. Top {k_top_units} Units Identified.")


    # 4. 個別可視化ループ
    
    # 4a. SAEユニットの可視化
    print("\n--- 4a. Visualizing Top SAE Units ---")
    for i in range(k_top_units):
        unit_idx = top_sae_indices[i].item()
        score = top_sae_scores[i].item()
        rank = i + 1 # 順位
        print(f"  -> {rank}. SAE Unit {unit_idx} (Score: {score:.6f})")
        
        visualize_single_unit('SAE', layer_idx, unit_idx, rank, vit_model, sae_model, dataloader_full, num_images_to_visualize, save_dir)

    # 4b. MAEユニットの可視化
    print("\n--- 4b. Visualizing Top MAE Units ---")
    for i in range(k_top_units):
        unit_idx = top_mae_indices[i].item()
        score = top_mae_scores[i].item()
        rank = i + 1 # 順位
        print(f"  -> {rank}. MAE Unit {unit_idx} (Score: {score:.6f})")
        
        visualize_single_unit('MAE', layer_idx, unit_idx, rank, vit_model, sae_model, dataloader_full, num_images_to_visualize, save_dir)
        
    print(f"\nAnalysis for Layer {layer_idx} finished. Results saved to {save_dir}")


if __name__ == "__main__":
    # 全層探索
    """
    for layer in LAYER_TO_ANALYZE:
        print(f"\n==========================================")
        print(f"STARTING ANALYSIS FOR LAYER {layer}")
        print(f"==========================================")
        analyze_and_visualize_top_k(layer, K_TOP_UNITS, NUM_IMAGES_TO_VISUALIZE)
    """
    # 単層探索
    layer = LAYER_TO_ANALYZE  
    print(f"\n==========================================")
    print(f"STARTING ANALYSIS FOR LAYER {layer}")
    print(f"==========================================")
    analyze_and_visualize_top_k(layer, K_TOP_UNITS, NUM_IMAGES_TO_VISUALIZE)