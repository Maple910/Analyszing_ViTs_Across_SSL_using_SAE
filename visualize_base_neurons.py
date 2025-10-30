import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from glob import glob
from data_loader import get_imagenet_val_dataloader
from config import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def visualize_base_neurons(layer_idx):
    """
    MAEの特定の層の指定されたニューロンが強く活性化した画像群を可視化し、保存します。
    """
    # configから可視化保存ディレクトリを取得
    sae_dir = SAE_VISUALIZE_DIRS[SAE_VISUALIZE_DIR_IDX]
    save_dir = os.path.join(VISUALIZATION_PATH, sae_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 1. モデルのロード
    vit_model = timm.create_model(MAE_MODEL_NAME, pretrained=True).to(DEVICE)
    vit_model.eval()

    # 2. データバッチのロード
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
    
    # 4. config.pyで指定されたニューロンIDを可視化
    neuron_to_visualize = NEURON_TO_VISUALIZE
    
    print(f"Visualizing specified base neuron {neuron_to_visualize} in layer {layer_idx}...")
    
    # 5. 特定したニューロンのトップ画像を可視化
    top_activations, top_indices = torch.topk(layer_output[:, :, neuron_to_visualize].flatten(), NUM_IMAGES_TO_VISUALIZE)
    image_indices = top_indices // layer_output.shape[1]
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                       std=[1/0.229, 1/0.224, 1/0.225])

    for i, img_idx in enumerate(image_indices):
        img_tensor = unnormalize(images[img_idx].cpu())
        img_tensor = img_tensor.clamp(0, 1)
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        axes[i].imshow(img)
        axes[i].set_title(f"Act: {top_activations[i].item():.4f}")
        axes[i].axis('off')
        
    fig.suptitle(f"Top-Activating Images for Base Neuron {neuron_to_visualize} in Layer {layer_idx}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"base_neuron_{neuron_to_visualize}_layer_{layer_idx}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_base_neurons(LAYER_TO_ANALYZE)