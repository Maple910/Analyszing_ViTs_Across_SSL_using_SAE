import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from sae_model import SparseAutoencoder
from data_loader import get_imagenet_val_dataloader
from config import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def visualize_top_patches(layer_idx, feature_to_visualize):
    """
    データセット全体から、指定されたSAEの特徴に最も強く反応した画像パッチを可視化し、保存します。
    """
    # configからSAE重みディレクトリと可視化保存ディレクトリを取得
    sae_dir = SAE_VISUALIZE_DIRS[SAE_VISUALIZE_DIR_IDX]
    sae_weight_path = os.path.join(BASE_DIR, sae_dir, f"sae_layer_{layer_idx}.pth")
    save_dir = os.path.join(VISUALIZATION_PATH, sae_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # モデルのロード
    vit_model = timm.create_model(MAE_MODEL_NAME, pretrained=True).to(DEVICE)
    sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))

    vit_model.eval()
    sae_model.eval()

    dataloader = get_imagenet_val_dataloader(DATASET_PATH, BATCH_SIZE, RANDOM_SEED)
    dataset = dataloader.dataset

    all_activations = []
    all_global_patch_indices = []

    print("Collecting activations across the entire dataset...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            
            activations = {}
            hook_handle = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(
                get_activation(f"layer_{layer_idx}", activations)
            )

            vit_model(images)
            hook_handle.remove()
            
            layer_output = activations[f"layer_{layer_idx}"]
            _, sae_features = sae_model(layer_output.view(-1, D_MODEL))
            sae_features = sae_features.view(images.shape[0], -1, D_SAE)
            
            patch_activations = sae_features[:, :, feature_to_visualize].flatten()
            all_activations.append(patch_activations.cpu())
            
            num_patches = sae_features.shape[1]
            global_image_indices = torch.arange(i * BATCH_SIZE, (i + 1) * BATCH_SIZE).unsqueeze(1).to(DEVICE)
            global_patch_indices = (global_image_indices * num_patches).repeat(1, num_patches) + torch.arange(num_patches).to(DEVICE)
            all_global_patch_indices.append(global_patch_indices.flatten().cpu())

    all_activations_tensor = torch.cat(all_activations, dim=0)
    all_global_patch_indices_tensor = torch.cat(all_global_patch_indices, dim=0)
    
    top_activations, top_indices = torch.topk(all_activations_tensor, NUM_IMAGES_TO_VISUALIZE)
    top_global_patch_indices = all_global_patch_indices_tensor[top_indices]
    
    num_patches_per_image = 197 
    top_global_image_indices = top_global_patch_indices // num_patches_per_image
    top_local_patch_indices = top_global_patch_indices % num_patches_per_image

    print(f"Visualizing top patches for SAE Feature {feature_to_visualize} in Layer {layer_idx}...")

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(NUM_IMAGES_TO_VISUALIZE):
        global_image_idx = top_global_image_indices[i].item()
        local_patch_idx = top_local_patch_indices[i].item()

        img_path = dataset.image_paths[global_image_idx]
        image = Image.open(img_path).convert('RGB')
        
        if local_patch_idx == 0:
            patch_img = Image.new('RGB', (16, 16), color = 'red')
        else:
            patch_h = 16
            patch_w = 16
            patch_x = ((local_patch_idx - 1) % 14) * patch_w
            patch_y = ((local_patch_idx - 1) // 14) * patch_h
            image_resized = image.resize((224, 224), Image.LANCZOS)
            patch_img = image_resized.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
        
        axes[i].imshow(patch_img)
        axes[i].set_title(f"Act: {top_activations[i].item():.4f}")
        axes[i].axis('off')
        
    fig.suptitle(f"Top-Activating Patches for SAE Feature {feature_to_visualize} in Layer {layer_idx}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"top_patches_feature_{feature_to_visualize}_layer_{layer_idx}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_top_patches(LAYER_TO_ANALYZE, FEATURE_TO_VISUALIZE)