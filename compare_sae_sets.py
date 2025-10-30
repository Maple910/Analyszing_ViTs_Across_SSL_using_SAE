# compare_sae_sets.py
import torch
import timm
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sae_model import SparseAutoencoder
from config import *

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def compare_sae_sets(layer_idx, neuron_to_visualize):
    """
    複数のSAE重みセット間で、指定されたMAEニューロンに最も類似したSAE特徴を比較します。
    """
    # MAEモデルのロード
    vit_model = timm.create_model(MAE_MODEL_NAME, pretrained=True).to(DEVICE)
    vit_model.eval()

    # 指定されたMAEニューロンの重みベクトルを取得
    with torch.no_grad():
        neuron_direction = vit_model.blocks[layer_idx].mlp.fc2.weight.data[:, neuron_to_visualize]

    comparison_results = {}

    print(f"Comparing SAE sets for MAE Neuron {neuron_to_visualize} in Layer {layer_idx}...")

    # configで定義されたすべてのSAE重みセットをループ
    for sae_dir in SAE_VISUALIZE_DIRS:
        sae_weight_path = os.path.join(BASE_DIR, sae_dir, f"sae_layer_{layer_idx}.pth")

        if not os.path.exists(sae_weight_path):
            print(f"Warning: SAE weights not found at {sae_weight_path}. Skipping.")
            continue

        # SAEモデルのロード
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
        sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))
        sae_model.eval()

        # MAEニューロンと各SAE特徴の重みとのコサイン類似度を計算
        with torch.no_grad():
            sae_decoder_weights = sae_model.decoder.weight.data.T
            similarities = F.cosine_similarity(neuron_direction.unsqueeze(0), sae_decoder_weights, dim=1)
            
            # 最も類似度の高いSAE特徴を見つける
            max_similarity, sae_feature_idx = torch.max(similarities, dim=0)
            
            # 結果を記録
            comparison_results[sae_dir] = {
                "similar_sae_feature_id": sae_feature_idx.item(),
                "similarity_score": max_similarity.item()
            }

    # 結果の表示
    if not comparison_results:
        print("No SAE sets were compared.")
        return

    print("\n--- Comparison Results ---")
    for sae_dir, result in comparison_results.items():
        print(f"SAE Set: {sae_dir}")
        print(f"  -> Most similar SAE Feature: {result['similar_sae_feature_id']}")
        print(f"  -> Cosine Similarity: {result['similarity_score']:.4f}")
    
    # 結果を棒グラフで可視化
    sae_sets = list(comparison_results.keys())
    similarity_scores = [result['similarity_score'] for result in comparison_results.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sae_sets, similarity_scores)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Similarity to MAE Neuron {neuron_to_visualize} in Layer {layer_idx}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_sae_sets(LAYER_TO_ANALYZE, NEURON_TO_VISUALIZE)