# 各層でTop-1を特定，可視化

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

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

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

def compare_attribute_feature(layer_idx, num_images_to_visualize=16):
    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)
    if not os.path.exists(sae_weight_path):
        print(f"Error: SAE weights not found at {sae_weight_path}. Check SAE_WEIGHTS_DIR in config_celeba.py.")
        return

    vit_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE)
    
    sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))

    vit_model.eval()
    sae_model.eval()

    dataloader_attr, dataloader_non_attr = get_celeba_attribute_loaders(
        CELEBA_IMG_DIR, CELEBA_ATTR_PATH, BATCH_SIZE, RANDOM_SEED, NUM_IMAGES_TO_SAMPLE
    )
    transform_viz = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_full = FullCelebADatasetForViz(CELEBA_IMG_DIR, CELEBA_ATTR_PATH, transform_viz)
    dataloader_full = DataLoader(dataset_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"--- 4. Identifying SAE Feature in Layer {layer_idx} ---")
    avg_activations_sae_attr = collect_avg_activations(dataloader_attr, layer_idx, vit_model, sae_model, 'SAE')
    avg_activations_sae_non_attr = collect_avg_activations(dataloader_non_attr, layer_idx, vit_model, sae_model, 'SAE')
    diff_score_sae = avg_activations_sae_attr - avg_activations_sae_non_attr
    top_scores_sae, top_indices_sae = torch.topk(diff_score_sae, k=1, dim=0)
    attr_feature_idx = top_indices_sae[0].item()
    print(f"Identified Specific SAE Feature: ID {attr_feature_idx} (Score: {top_scores_sae[0].item():.6f})")

    print("--- 5. Identifying attribute MAE Neuron by Activation Difference ---")
    avg_activations_mae_attr = collect_avg_activations(dataloader_attr, layer_idx, vit_model, sae_model, 'MAE')
    avg_activations_mae_non_attr = collect_avg_activations(dataloader_non_attr, layer_idx, vit_model, sae_model, 'MAE')
    diff_score_mae = avg_activations_mae_attr - avg_activations_mae_non_attr
    top_scores_mae, top_indices_mae = torch.topk(diff_score_mae, k=1, dim=0)
    mae_neuron_idx = top_indices_mae[0].item()
    print(f"Identified Specific MAE Neuron: ID {mae_neuron_idx} (Score: {top_scores_mae[0].item():.6f})")

    all_activations_neuron = []
    all_activations_sae = []
    all_image_paths = []
    
    print("--- 6. Collecting Global Max Activations for Comparison ---")
    with torch.no_grad():
        for images, paths in tqdm(dataloader_full, desc="Global Max Act"):
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
            
            hidden_output_pre_act = activations[f"layer_{layer_idx}_fc1"]
            max_act_neuron, _ = torch.max(hidden_output_pre_act[:, :, mae_neuron_idx].abs(), dim=1)
            all_activations_neuron.append(max_act_neuron.cpu())
            
            layer_output = activations[f"layer_{layer_idx}_fc2"].view(-1, D_MODEL)
            _, sae_features = sae_model(layer_output)
            sae_features = sae_features.view(images.shape[0], -1, D_SAE)
            max_act_sae, _ = torch.max(sae_features[:, :, attr_feature_idx], dim=1)
            all_activations_sae.append(max_act_sae.cpu())
            
            all_image_paths.extend(paths)

    all_activations_neuron_tensor = torch.cat(all_activations_neuron, dim=0)
    total_images = all_activations_neuron_tensor.size(0)
    k = min(num_images_to_visualize, total_images)
    _, top_idx_neuron = torch.topk(all_activations_neuron_tensor, k=k)
    
    all_activations_sae_tensor = torch.cat(all_activations_sae, dim=0)
    _, top_idx_sae = torch.topk(all_activations_sae_tensor, k=k)

    ncols = int(np.ceil(k / 4.0))
    fig, axes = plt.subplots(4, ncols, figsize=(15, 12))
    axes = axes.flatten()
    print("--- 7. Generating Comparison Grid ---")
    k_half = k // 2

    for i in range(k_half):
        global_idx = top_idx_neuron[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[i].imshow(image_transformed)
        axes[i].set_title(f"MAE (N{mae_neuron_idx})", fontsize=8)
        axes[i].axis('off')

    for i in range(k_half):
        global_idx = top_idx_sae[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[i + k_half].imshow(image_transformed)
        axes[i + k_half].set_title(f"SAE (F{attr_feature_idx})", fontsize=8)
        axes[i + k_half].axis('off')

    for j in range(k, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Layer {layer_idx} Comparison: MAE Neuron {mae_neuron_idx} vs. SAE Feature {attr_feature_idx}")
    plt.tight_layout()

    save_path = os.path.join(ANALYSIS_PATH, f"{TARGET_ATTRIBUTE}_comparison_Layer_{layer_idx}.png")
    plt.savefig(save_path)
    print(f"\nVisualization saved to {save_path}")
    plt.close('all')

if __name__ == "__main__":
    for layer in range(12):
        print(f"\n=== Analyzing Layer {layer} ===")
        compare_attribute_feature(layer, NUM_IMAGES_TO_VISUALIZE)