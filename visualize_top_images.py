import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from glob import glob
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

def visualize_top_images(layer_idx, feature_to_visualize):
    """
    指定されたSAEの特徴が最も強く反応した画像N枚を可視化し、保存します。
    （同じ画像が複数表示される問題を回避）
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

    # 単一のデータバッチをロード
    dataloader = get_imagenet_val_dataloader(DATASET_PATH, BATCH_SIZE, RANDOM_SEED)
    images, _ = next(iter(dataloader))
    images = images.to(DEVICE)
    
    # MAEのアクティベーションを取得
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
    
    max_activations_per_image, _ = torch.max(sae_features[:, :, feature_to_visualize], dim=1)
    
    top_activations, top_indices = torch.topk(max_activations_per_image, NUM_IMAGES_TO_VISUALIZE)
    
    print(f"Visualizing images for SAE Feature {feature_to_visualize} in Layer {layer_idx}...")
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                       std=[1/0.229, 1/0.224, 1/0.225])

    for i, img_idx in enumerate(top_indices):
        img_tensor = unnormalize(images[img_idx].cpu())
        img_tensor = img_tensor.clamp(0, 1)
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        axes[i].imshow(img)
        axes[i].set_title(f"Act: {top_activations[i].item():.4f}")
        axes[i].axis('off')
        
    fig.suptitle(f"Top-Activating Images for SAE Feature {feature_to_visualize} in Layer {layer_idx}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"top_images_feature_{feature_to_visualize}_layer_{layer_idx}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_top_images(LAYER_TO_ANALYZE, FEATURE_TO_VISUALIZE)