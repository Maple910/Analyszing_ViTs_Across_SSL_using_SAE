# compare_blond_feature.py (最終版: MAEニューロンを差分で特定)

import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from tqdm import tqdm
from sae_model_debug import SparseAutoencoderDebug as SparseAutoencoder
from data_loader_celeba import get_celeba_attribute_loaders
from config_celeba import * 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Hook関数
def get_activation(name, activations):
    def hook(model, input, output):
        # GELU/活性化関数適用前のテンソルをそのまま取得
        activations[name] = output.detach()
    return hook

# 全てのCelebA画像から活性化を収集するためのデータセット
class FullCelebADatasetForViz(Dataset):
    def __init__(self, img_dir, attr_path, transform):
        # 画像ファイルパスのリストを作成
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

# MAE/SAEの活性化を収集するユーティリティ関数
def collect_avg_activations(dataloader, layer_idx, vit_model, sae_model, target_type):
    """
    指定されたデータローダーに対して、SAE特徴またはMAEニューロンの平均活性化を収集する。
    target_type: 'SAE' または 'MAE'
    """
    # MAE の MLP 隠れ次元（通常 D_MODEL * 4）
    D_MLP = D_MODEL * 4

    # SAE の場合は D_SAE、MAE の場合は D_MLP を使う
    sum_activations = torch.zeros(D_SAE if target_type == 'SAE' else D_MLP).to(DEVICE)
    patch_count = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            activations = {}
            
            # MAEの活性化をフック
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
            if patch_count == 0:
                lo = layer_output.detach().cpu()
                print(f"[DEBUG] layer_output shape: {layer_output.shape}, stats: min={lo.min().item():.6e}, max={lo.max().item():.6e}, mean={lo.mean().item():.6e}, std={lo.std().item():.6e}")
                fc2_raw = activations[f"layer_{layer_idx}_fc2"].detach().cpu()
                print(f"[DEBUG] raw fc2 per-batch shape: {fc2_raw.shape}, raw fc2 stats: min={fc2_raw.min().item():.6e}, max={fc2_raw.max().item():.6e}, mean={fc2_raw.mean().item():.6e}")
                # 追加デバッグ: SAE に実際に入力してみる（同じデバイス/型で）
                sample = layer_output[:256]  # small subset
                print(f"[DEBUG] sample device/dtype: {sample.device}, {sample.dtype}")
                try:
                    # debug: スパース化をバイパスして pre_act を直接取得
                    rec_s, feat_s = sae_model(sample, debug=True, use_leaky=True, bypass_shrink=True)
                    fcpu = feat_s.detach().cpu()
                    print(f"[DEBUG] SAE on sample feat shape: {feat_s.shape}, min={fcpu.min():.6e}, max={fcpu.max():.6e}, mean={fcpu.mean():.6e}, std={fcpu.std():.6e}")
                    print(f"[DEBUG] SAE feat zero fraction: {(fcpu==0).float().mean().item():.6f}")
                except Exception as e:
                     print(f"[DEBUG] SAE forward(sample) raised: {e}")
                # パラメータ規模を簡易表示
                with torch.no_grad():
                    enc_w = sae_model.encoder.weight.detach().cpu()
                    enc_b = sae_model.encoder.bias.detach().cpu() if hasattr(sae_model.encoder, 'bias') else None
                    print(f"[DEBUG] encoder.weight abs mean: {enc_w.abs().mean().item():.6e}, encoder.weight max abs: {enc_w.abs().max().item():.6e}")
                    if enc_b is not None:
                        print(f"[DEBUG] encoder.bias sample: {enc_b.view(-1)[:10].numpy()}")
            if target_type == 'SAE':
                _, features = sae_model(layer_output, debug=True)
                # --- DEBUG: 最初のバッチだけ形状と統計を表示 ---
                if patch_count == 0:
                    print(f"[DEBUG] SAE features shape (batch*tokens x D_SAE): {features.shape}")
                    f_cpu = features.detach().cpu()
                    print(f"[DEBUG] SAE features stats: min={f_cpu.min():.6f}, max={f_cpu.max():.6f}, mean={f_cpu.mean():.6f}, std={f_cpu.std():.6f}")
                    # 0 の割合
                    zero_frac = (f_cpu == 0).float().mean().item()
                    print(f"[DEBUG] SAE features zero fraction: {zero_frac:.6f}")
                # optional: 絶対値で集計する場合は features = features.abs()
            else: # target_type == 'MAE'
                # MAEニューロンの活性化（プリ・アクティベーション）
                features = activations[f"layer_{layer_idx}_fc1"].view(-1, D_MLP) # MLP 次元に合わせる
                # 活性化の強さの指標として絶対値の最大値(|x|)の平均を計算
                features = features.abs()

            sum_activations.add_(features.sum(dim=0))
            patch_count += features.shape[0]
            
    return sum_activations / patch_count


def compare_blond_feature(layer_idx, num_images_to_visualize=16):
    
    # 1. パスの設定とディレクトリ作成
    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    sae_weight_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)

    if not os.path.exists(sae_weight_path):
        print(f"Error: SAE weights not found at {sae_weight_path}. Check SAE_WEIGHTS_DIR in config_celeba.py.")
        return

    # 2. モデルのロード
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE)
    # デバッグ版でも shrink_lambda は使わない
    sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_weight_path, map_location=DEVICE))

    vit_model.eval()
    sae_model.eval()

    # --- DEBUG: SAE 重みの簡易チェック ---
    total_param_sum = 0.0
    for p in sae_model.parameters():
        total_param_sum += float(p.detach().abs().sum().item())
    print(f"[DEBUG] SAE total abs param sum: {total_param_sum:.6e}")
    nonzero_counts = {name: (p.detach().cpu().abs() > 0).sum().item() for name, p in sae_model.named_parameters()}
    for n, c in list(nonzero_counts.items())[:6]:
        print(f"[DEBUG] param {n} nonzero_count={c}")
    print("[DEBUG] SAE state_dict keys (sample):", list(sae_model.state_dict().keys())[:8])

    # --- DEBUG: SAE 単体の応答確認（ランダム入力） ---
    with torch.no_grad():
        test_in = torch.randn(2, D_MODEL).to(DEVICE)
        try:
            rec, feat = sae_model(test_in, debug=True)
            print(f"[DEBUG] SAE random-input feat shape: {feat.shape}, stats: min={feat.min().item():.6e}, max={feat.max().item():.6e}, mean={feat.mean().item():.6e}")
        except Exception as e:
            print(f"[DEBUG] SAE forward on random input failed: {e}")

    # 3. データローダーのロード
    dataloader_blond, dataloader_non_blond = get_celeba_attribute_loaders(
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
    
    
    # 4. 特徴特定フェーズ: SAE特徴の特定
    print(f"--- 4. Identifying Blond Feature in Layer {layer_idx} ---")
    
    # SAE活性化の平均を収集
    avg_activations_sae_blond = collect_avg_activations(dataloader_blond, layer_idx, vit_model, sae_model, 'SAE')
    avg_activations_sae_non_blond = collect_avg_activations(dataloader_non_blond, layer_idx, vit_model, sae_model, 'SAE')
    
    # 差分スコアの計算とトップ特徴の特定
    diff_score_sae = avg_activations_sae_blond - avg_activations_sae_non_blond
    top_scores_sae, top_indices_sae = torch.topk(diff_score_sae, k=1, dim=0)
    blond_feature_idx = top_indices_sae[0].item()
    print(f"🥇 Identified Blond-Specific SAE Feature: ID {blond_feature_idx} (Score: {top_scores_sae[0].item():.6f})")

    
    # 5. MAEニューロンの特定フェーズ: 活性化の差分でトップのMAEニューロンを探す
    print("--- 5. Identifying Blond MAE Neuron by Activation Difference ---")
    
    # MAE活性化の平均を収集
    avg_activations_mae_blond = collect_avg_activations(dataloader_blond, layer_idx, vit_model, sae_model, 'MAE')
    avg_activations_mae_non_blond = collect_avg_activations(dataloader_non_blond, layer_idx, vit_model, sae_model, 'MAE')
    
    # 差分スコアの計算とトップニューロンの特定
    diff_score_mae = avg_activations_mae_blond - avg_activations_mae_non_blond
    top_scores_mae, top_indices_mae = torch.topk(diff_score_mae, k=1, dim=0)
    
    # MLP隠れ層の次元は D_MODEL * 4 ですが、ここではD_MODEL=768 なので 768 * 4 = 3072
    mae_neuron_idx = top_indices_mae[0].item()
    print(f"🥇 Identified Blond-Specific MAE Neuron: ID {mae_neuron_idx} (Score: {top_scores_mae[0].item():.6f})")

    # 6. グローバル活性化収集フェーズ: 両方のトップk画像を特定
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
            
            # MAE Neuronの活性化
            hidden_output_pre_act = activations[f"layer_{layer_idx}_fc1"]
            # 活性化の強さの指標として絶対値の最大値を取る
            max_act_neuron, _ = torch.max(hidden_output_pre_act[:, :, mae_neuron_idx].abs(), dim=1)
            all_activations_neuron.append(max_act_neuron.cpu())
            
            # SAE Featureの活性化
            layer_output = activations[f"layer_{layer_idx}_fc2"].view(-1, D_MODEL) 
            _, sae_features = sae_model(layer_output, debug=True)
            sae_features = sae_features.view(images.shape[0], -1, D_SAE)
            max_act_sae, _ = torch.max(sae_features[:, :, blond_feature_idx], dim=1)
            all_activations_sae.append(max_act_sae.cpu())
            
            all_image_paths.extend(paths)

    # トップk画像のインデックスをグローバルに特定
    all_activations_neuron_tensor = torch.cat(all_activations_neuron, dim=0)
    total_images = all_activations_neuron_tensor.size(0)
    k = min(num_images_to_visualize, total_images)

    _, top_idx_neuron = torch.topk(all_activations_neuron_tensor, k=k)
    
    all_activations_sae_tensor = torch.cat(all_activations_sae, dim=0)
    _, top_idx_sae = torch.topk(all_activations_sae_tensor, k=k)

    # 7. 比較可視化
    unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                       std=[1/0.229, 1/0.224, 1/0.225])

    ncols = int(np.ceil(k / 4.0))
    fig, axes = plt.subplots(4, ncols, figsize=(15, 12))
    axes = axes.flatten()
    print("--- 7. Generating Comparison Grid ---")
    k_half = k // 2

    # 上半分: MAE Neuron のトップ画像
    for i in range(k_half):
        global_idx = top_idx_neuron[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')

        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[i].imshow(image_transformed)
        axes[i].set_title(f"MAE (N{mae_neuron_idx})", fontsize=8)
        axes[i].axis('off')

    # 下半分: SAE Feature のトップ画像
    for i in range(k_half):
        global_idx = top_idx_sae[i].item()
        img_path = all_image_paths[global_idx]
        image = Image.open(img_path).convert('RGB')

        image_transformed = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
        axes[i + k_half].imshow(image_transformed)
        axes[i + k_half].set_title(f"SAE (F{blond_feature_idx})", fontsize=8)
        axes[i + k_half].axis('off')

    # 余った軸は非表示にする
    for j in range(k, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Layer {layer_idx} Comparison: MAE Neuron {mae_neuron_idx} vs. Blond Feature {blond_feature_idx}")
    plt.tight_layout()

    save_path = os.path.join(ANALYSIS_PATH, f"blond_comparison_Layer_{layer_idx}_debug.png")
    plt.savefig(save_path)
    print(f"\nVisualization saved to {save_path}")
    plt.close('all')

if __name__ == "__main__":
    for layer in range(12):
        print(f"\n=== Analyzing Layer {layer} ===")
        compare_blond_feature(layer, NUM_IMAGES_TO_VISUALIZE)