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

def visualize_top_patches_in_batch(layer_idx, feature_to_visualize):
    """
    単一のデータバッチ内で、指定されたSAEの特徴に最も強く反応した画像パッチを可視化し、保存します。
    """
    # configからSAE重みディレクトリと可視化保存ディレクトリを取得
    sae_dir = SAE_VISUALIZE_DIRS[SAE_VISUALIZE_DIR_IDX]
    sae_weight_path = os.path.join(BASE_DIR, sae_dir, f"sae_layer_{layer_idx}.pth")
    save_dir = os.path.join(VISUALIZATION_PATH, sae_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. モデルのロード
    vit_model = timm.create_model(MAE_MODEL_NAME, pretrained=True).to(DEVICE)
    sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))

    vit_model.eval()
    sae_model.eval()

    # 2. 単一のデータバッチをロード
    dataloader = get_imagenet_val_dataloader(DATASET_PATH, BATCH_SIZE, RANDOM_SEED)
    images, _ = next(iter(dataloader))
    images = images.to(DEVICE)

    # 3. MAEのアクティベーションを取得
    activations = {}
    hook_handle = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(
        get_activation(f"layer_{layer_idx}", activations)
    )

    with torch.no_grad():
        vit_model(images)
    hook_handle.remove()
    
    layer_output = activations[f"layer_{layer_idx}"]
    _, sae_features = sae_model(layer_output.view(-1, D_MODEL))
    
    sae_features = sae_features.view(images.shape[0], -1, D_SAE)
    
    # 4. バッチ内の各パッチのアクティベーションを収集
    patch_activations = sae_features[:, :, feature_to_visualize].flatten()
    
    top_activations, top_indices = torch.topk(patch_activations, NUM_IMAGES_TO_VISUALIZE)
    
    num_patches_per_image = 197
    top_local_patch_indices = top_indices % num_patches_per_image
    top_local_image_indices = top_indices // num_patches_per_image

    print(f"Visualizing top patches in batch for SAE Feature {feature_to_visualize} in Layer {layer_idx}...")

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(NUM_IMAGES_TO_VISUALIZE):
        local_image_idx = top_local_image_indices[i].item()
        local_patch_idx = top_local_patch_indices[i].item()

        image = images[local_image_idx].cpu()
        
        if local_patch_idx == 0:
            patch_img = Image.new('RGB', (16, 16), color = 'red')
        else:
            unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                               std=[1/0.229, 1/0.224, 1/0.225])
            image_tensor = unnormalize(image).clamp(0, 1)
            image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            
            patch_h = 16
            patch_w = 16
            patch_x = ((local_patch_idx - 1) % 14) * patch_w
            patch_y = ((local_patch_idx - 1) // 14) * patch_h
            patch_img = image_pil.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
        
        axes[i].imshow(patch_img)
        axes[i].set_title(f"Act: {top_activations[i].item():.4f}")
        axes[i].axis('off')
        
    fig.suptitle(f"Top-Activating Patches in Batch for SAE Feature {feature_to_visualize} in Layer {layer_idx}")
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"top_patches_in_batch_feature_{feature_to_visualize}_layer_{layer_idx}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_top_patches_in_batch(LAYER_TO_ANALYZE, FEATURE_TO_VISUALIZE)