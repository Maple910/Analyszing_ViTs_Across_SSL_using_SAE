# analyze_unit_statistics.py
# デコーダ重みの類似度，活性化の相関ヒートマップの作成
# デフォルトはTOP10特徴間

import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn.functional as F
from tqdm import tqdm
from sae_model import SparseAutoencoder
from data_loader_celeba import get_celeba_attribute_loaders
from config_celeba import * 
from torch.utils.data import DataLoader

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 統計分析を行う関数
def analyze_unit_statistics(layer_idx, top_k=10):
    print(f"\n=== Analyzing Unit Statistics for Layer {layer_idx} (Top {top_k} Units) ===")
    
    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)

    if not os.path.exists(sae_weight_path):
        print(f"Error: SAE weights not found at {sae_weight_path}.")
        return

    # 1. モデルとデータの準備
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))
    vit_model.eval()
    sae_model.eval()

    # 属性あり/なしデータローダー
    # 修正: target_attribute 引数を削除
    dataloader_attr, dataloader_non_attr = get_celeba_attribute_loaders(
        CELEBA_IMG_DIR, CELEBA_ATTR_PATH, BATCH_SIZE, RANDOM_SEED, NUM_IMAGES_TO_SAMPLE
    )

    # 2. トップ特徴の特定 (平均活性化の差分に基づく)
    print("--- Identifying Top Attribute-Specific Features ---")
    
    def get_avg_acts(dataloader):
        sum_acts = torch.zeros(D_SAE).to(DEVICE)
        count = 0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(DEVICE)
                activations = {}
                hook = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(
                    get_activation("fc2", activations)
                )
                vit_model(images)
                hook.remove()
                
                layer_output = activations["fc2"].view(-1, D_MODEL)
                _, features = sae_model(layer_output)
                sum_acts += features.sum(dim=0)
                count += features.shape[0]
        return sum_acts / count

    avg_acts_attr = get_avg_acts(dataloader_attr)
    avg_acts_non_attr = get_avg_acts(dataloader_non_attr)
    
    diff_scores = avg_acts_attr - avg_acts_non_attr
    
    # トップKの特徴を取得
    top_scores, top_indices = torch.topk(diff_scores, k=top_k)
    top_indices_cpu = top_indices.cpu().numpy()
    
    print(f"Top {top_k} Feature IDs: {top_indices_cpu}")
    print(f"Scores: {top_scores.cpu().numpy()}")

    # 3. 統計量の計算
    print("\n--- Calculating Statistics (Similarity & Co-activation) ---")

    # A. デコーダ重みのコサイン類似度
    decoder_weights = sae_model.decoder.weight.data.T # (D_SAE, D_MODEL)
    top_weights = decoder_weights[top_indices] # (K, D_MODEL)
    
    # 正規化して内積（コサイン類似度）をとる
    top_weights_norm = F.normalize(top_weights, p=2, dim=1)
    cosine_sim_matrix = torch.mm(top_weights_norm, top_weights_norm.T).cpu().numpy()

    # B. 活性化の共起（相関行列）
    # 属性ありデータセットにおけるトップ特徴の活性化系列を収集
    activations_list = []
    
    print("Collecting activations for correlation analysis...")
    with torch.no_grad():
        for images, _ in dataloader_attr: 
            images = images.to(DEVICE)
            activations = {}
            hook = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(
                get_activation("fc2", activations)
            )
            vit_model(images)
            hook.remove()
            
            layer_output = activations["fc2"].view(-1, D_MODEL)
            _, features = sae_model(layer_output)
            # (Batch*Patches, D_SAE) -> トップ特徴のみ抽出
            top_features_acts = features[:, top_indices] 
            activations_list.append(top_features_acts.cpu())
            
    all_acts = torch.cat(activations_list, dim=0).numpy() # (N_samples, K)
    
    # 相関行列の計算 (numpy.corrcoef)
    correlation_matrix = np.corrcoef(all_acts.T)
    # NaN対策
    correlation_matrix = np.nan_to_num(correlation_matrix)

    # 4. 可視化と保存
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # コサイン類似度ヒートマップ
    sns.heatmap(cosine_sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=top_indices_cpu, yticklabels=top_indices_cpu, ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title(f"Cosine Similarity of Decoder Weights\n(Layer {layer_idx}, Top {top_k} {TARGET_ATTRIBUTE} Features)")
    
    # 共起相関ヒートマップ
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=top_indices_cpu, yticklabels=top_indices_cpu, ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title(f"Activation Correlation (Co-occurrence)\n(Layer {layer_idx}, Top {top_k} {TARGET_ATTRIBUTE} Features)")
    
    plt.tight_layout()
    save_path = os.path.join(ANALYSIS_PATH, f"{TARGET_ATTRIBUTE}_stats_layer_{layer_idx}_top{top_k}.png")
    plt.savefig(save_path)
    print(f"Statistics visualization saved to {save_path}")
    plt.close('all')

if __name__ == "__main__":
    #トップ10特徴間分析
    layer = LAYER_TO_ANALYZE 
    #LAYER_TO_ANALYZE がリストならループ処理に変更
    if isinstance(layer, list):
        for l in layer:
            analyze_unit_statistics(l, top_k=10)
    else:
        analyze_unit_statistics(layer, top_k=10)