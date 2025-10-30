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

def visualize_top_globally_comparison(layer_idx, neuron_to_visualize):
    """
    データセット全体から、指定されたMAEニューロンと、
    最も類似したSAEの特徴に反応した画像を比較可視化し、保存します。
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

    # データセットとデータローダーをロード
    dataloader = get_imagenet_val_dataloader(DATASET_PATH, BATCH_SIZE, RANDOM_SEED)
    dataset = dataloader.dataset

    # 1. 最も類似したSAE特徴を見つける
    with torch.no_grad():
        neuron_direction = vit_model.blocks[layer_idx].mlp.fc2.weight.data[:, neuron_to_visualize]
        sae_decoder_weights = sae_model.decoder.weight.data.T
        similarities = F.cosine_similarity(neuron_direction.unsqueeze(0), sae_decoder_weights, dim=1)
        sae_feature_idx = torch.argmax(similarities).item()
    
    # 2. MAEニューロンとSAE特徴のアクティベーションを全データセットで集計
    all_activations_neuron = []
    all_activations_sae = []
    
    print("Collecting activations for comparison across the entire dataset...")
    
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
            
            max_act_neuron, _ = torch.max(layer_output[:, :, neuron_to_visualize], dim=1)
            all_activations_neuron.append(max_act_neuron.cpu())

            max_act_sae, _ = torch.max(sae_features[:, :, sae_feature_idx], dim=1)
            all_activations_sae.append(max_act_sae.cpu())

    # 3. 上位N枚の画像を特定
    all_activations_neuron_tensor = torch.cat(all_activations_neuron, dim=0)
    top_act_neuron, top_idx_neuron = torch.topk(all_activations_neuron_tensor, NUM_IMAGES_TO_VISUALIZE)
    
    all_activations_sae_tensor = torch.cat(all_activations_sae, dim=0)
    top_act_sae, top_idx_sae = torch.topk(all_activations_sae_tensor, NUM_IMAGES_TO_VISUALIZE)

    # 4. 可視化
    fig, axes = plt.subplots(2, NUM_IMAGES_TO_VISUALIZE // 2, figsize=(15, 6))
    axes = axes.flatten()

    unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                       std=[1/0.229, 1/0.224, 1/0.225])

    print(f"Visualizing comparison for Neuron {neuron_to_visualize} vs. SAE Feature {sae_feature_idx}...")
    
    for i in range(NUM_IMAGES_TO_VISUALIZE // 2):
        global_idx = top_idx_neuron[i]
        img_path = dataset.image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])(image)
        img_tensor = unnormalize(image_transformed)
        img_tensor = img_tensor.clamp(0, 1)
        axes[i].imshow(Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
        axes[i].set_title(f"N Act: {top_act_neuron[i].item():.4f}")
        axes[i].axis('off')
        
    for i in range(NUM_IMAGES_TO_VISUALIZE // 2):
        global_idx = top_idx_sae[i]
        img_path = dataset.image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')
        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])(image)
        img_tensor = unnormalize(image_transformed)
        img_tensor = img_tensor.clamp(0, 1)
        axes[i + NUM_IMAGES_TO_VISUALIZE // 2].imshow(Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
        axes[i + NUM_IMAGES_TO_VISUALIZE // 2].set_title(f"S Act: {top_act_sae[i].item():.4f}")
        axes[i + NUM_IMAGES_TO_VISUALIZE // 2].axis('off')

    fig.suptitle(f"Layer {layer_idx} Comparison: MAE Neuron {neuron_to_visualize} vs. Similar SAE Feature {sae_feature_idx}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"comparison_n{neuron_to_visualize}_vs_s{sae_feature_idx}_layer_{layer_idx}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()
    
if __name__ == "__main__":
    visualize_top_globally_comparison(LAYER_TO_ANALYZE, NEURON_TO_VISUALIZE)