# list_top_features.py
import torch
import timm
import numpy as np
import os
from sae_model import SparseAutoencoder
from data_loader import get_imagenet_val_dataloader
from config import *

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def list_top_features(layer_idx, top_n=10):
    sae_dir = SAE_VISUALIZE_DIRS[SAE_VISUALIZE_DIR_IDX]
    sae_weight_path = os.path.join(BASE_DIR, sae_dir, f"sae_layer_{layer_idx}.pth")
    save_dir = os.path.join(VISUALIZATION_PATH, sae_dir)
    os.makedirs(save_dir, exist_ok=True)

    vit_model = timm.create_model(MAE_MODEL_NAME, pretrained=True).to(DEVICE)
    sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))

    vit_model.eval()
    sae_model.eval()

    dataloader = get_imagenet_val_dataloader(DATASET_PATH, BATCH_SIZE, RANDOM_SEED)
    images, _ = next(iter(dataloader))
    images = images.to(DEVICE)
    
    activations = {}
    hook_handle = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(
        get_activation(f"layer_{layer_idx}", activations)
    )

    with torch.no_grad():
        vit_model(images)
    hook_handle.remove()
    
    layer_output = activations[f"layer_{layer_idx}"]
    _, sae_features = sae_model(layer_output.view(-1, D_MODEL))
    
    max_activations_per_feature, _ = torch.max(sae_features, dim=0)

    sorted_activations, sorted_indices = torch.sort(max_activations_per_feature, descending=True)
    
    output_filename = f"top_features_layer_{layer_idx}.txt"
    output_path = os.path.join(save_dir, output_filename)
    with open(output_path, "w") as f:
        f.write(f"Analyzing SAE features in layer {layer_idx}...\n")
        f.write(f"Top {top_n} activating features:\n")
        
        print(f"Analyzing SAE features in layer {layer_idx}...")
        print(f"Top {top_n} activating features:")

        for i in range(top_n):
            feature_id = sorted_indices[i].item()
            activation_value = sorted_activations[i].item()
            
            if activation_value > 0:
                line = f"{i+1}. Feature ID: {feature_id}, Max Activation: {activation_value:.4f}"
                print(line)
                f.write(line + "\n")
            else:
                line = "All remaining features have activation 0."
                print(line)
                f.write(line + "\n")
                break
    
    print(f"\nResults have been saved to {output_path}")

if __name__ == "__main__":
    list_top_features(LAYER_TO_ANALYZE)