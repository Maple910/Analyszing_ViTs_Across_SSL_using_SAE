import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from data_loader import get_imagenet_val_dataloader
from config import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def visualize_top_globally_neurons(layer_idx, neuron_to_visualize):
    """
    データセット全体から、指定されたMAEのニューロンに最も強く反応した画像を可視化し、保存します。
    """
    # configから可視化保存ディレクトリを取得
    sae_dir = SAE_VISUALIZE_DIRS[SAE_VISUALIZE_DIR_IDX]
    save_dir = os.path.join(VISUALIZATION_PATH, sae_dir)
    os.makedirs(save_dir, exist_ok=True)

    # モデルのロード
    vit_model = timm.create_model(MAE_MODEL_NAME, pretrained=True).to(DEVICE)
    vit_model.eval()

    # データセットとデータローダーをロード
    dataloader = get_imagenet_val_dataloader(DATASET_PATH, BATCH_SIZE, RANDOM_SEED)
    dataset = dataloader.dataset

    all_activations = []
    
    print("Collecting MAE neuron activations across the entire dataset...")
    
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
            
            max_activations_per_image, _ = torch.max(layer_output[:, :, neuron_to_visualize], dim=1)
            all_activations.append(max_activations_per_image.cpu())

    all_activations_tensor = torch.cat(all_activations, dim=0)
    top_activations, top_indices_global = torch.topk(all_activations_tensor, NUM_IMAGES_TO_VISUALIZE)
    
    print(f"Visualizing top images for MAE Neuron {neuron_to_visualize} in Layer {layer_idx}...")

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    # 逆正規化の処理
    unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                       std=[1/0.229, 1/0.224, 1/0.225])

    for i, global_idx in enumerate(top_indices_global):
        img_path = dataset.image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        
        image_transformed = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])(image)

        img_tensor = unnormalize(image_transformed)
        img_tensor = img_tensor.clamp(0, 1) # 値を0から1の範囲にクリップ
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        axes[i].imshow(img)
        axes[i].set_title(f"Act: {top_activations[i].item():.4f}")
        axes[i].axis('off')
        
    fig.suptitle(f"Top-Activating Images for MAE Neuron {neuron_to_visualize} in Layer {layer_idx}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"mae_neuron_{neuron_to_visualize}_layer_{layer_idx}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_top_globally_neurons(LAYER_TO_ANALYZE, NEURON_TO_VISUALIZE)