# MAEとMoCoとでヒートマップ比較

import os
import re
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms

# SAEモデル定義と設定ファイルのインポート
from sae_model import SparseAutoencoder
import config_oid as cfg_mae  # MAE用の設定
import config_moco as cfg_moco # MoCo用の設定

# ==========================================
# ★設定項目（ここを環境に合わせて書き換えてください）
# ==========================================
# 1. 比較したい属性名
TARGET_ATTRIBUTE = "Guitar"

# 2. MAEの分析結果があるディレクトリパス（stats_full.txtがある場所）
# 例: "./data/analysis_oid/analysis_results_oid_Car/run_0" など、
MAE_RESULTS_DIR = f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11"

# 3. MoCoの分析結果があるディレクトリパス
# 例: "./data/analysis_moco/analysis_results_moco_Car/run_0" など
MOCO_RESULTS_DIR = f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"

# 4. 比較画像の保存先ディレクトリ
COMPARISON_SAVE_DIR = "./data/analysis_comparison_heatmaps/mae_run_9_moco_run_1_normalize"
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_sae_info(txt_path, model_type):
    """テキストファイルからSAEのベスト層とユニットIDを抽出"""
    if not os.path.exists(txt_path):
        print(f"[ERROR] File not found: {txt_path}")
        return None, None
    with open(txt_path, 'r') as f: content = f.read()
    
    # 検索パターン (MoCoは表記が揺れる可能性があるため複数対応)
    if model_type == 'MAE':
        pattern = r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)"
    else:
        pattern = r"GLOBAL BEST SAE \(MoCo\): Layer (\d+), Unit (\d+)"
        if not re.search(pattern, content):
            pattern = r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)"

    match = re.search(pattern, content)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        print(f"[ERROR] Could not find SAE info in {txt_path} for {model_type}")
        return None, None

def load_moco_weights(model):
    """MoCo v3の重みをロードするヘルパー関数"""
    url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    new_state_dict = {k.replace("module.base_encoder.", ""): v for k, v in state_dict.items() if k.startswith("module.base_encoder.")}
    model.load_state_dict(new_state_dict, strict=False)
    return model

def create_heatmap(model, sae, img_tensor, layer, unit_id, hook_name):
    """指定されたモデルとSAEを使ってヒートマップを作成"""
    activations = {}
    def hook(m, i, o): activations[hook_name] = o.detach()
    
    # モデル固有のフック場所を指定
    if hook_name == 'mae_fc2':
        handle = model.blocks[layer].mlp.fc2.register_forward_hook(hook)
        feats_idx = 1 # CLSトークンを除外
    else: # moco_fc2
        handle = model.blocks[layer].mlp.fc2.register_forward_hook(hook)
        feats_idx = 1

    with torch.no_grad():
        model(img_tensor)
    handle.remove()
    
    # 中間層出力を取得 -> SAEに入力
    raw_feats = activations[hook_name][:, feats_idx:, :] # (1, 196, D)
    B, N, D = raw_feats.shape
    _, sae_flat = sae(raw_feats.reshape(-1, D))
    sae_feats = sae_flat.reshape(B, N, -1)
    
    # 特定ユニットのマップを抽出して整形
    target_map = sae_feats[0, :, unit_id]
    score = target_map.max().item()
    hm = target_map.reshape(14, 14).detach().cpu().numpy()
    hm = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_CUBIC)
    hm = np.maximum(hm, 0)
    if hm.max() > 0: hm /= hm.max()
    
    return hm, score

def main():
    print(f"=== Creating Comparison Heatmap for attribute: '{TARGET_ATTRIBUTE}' ===")
    os.makedirs(COMPARISON_SAVE_DIR, exist_ok=True)

    # 1. 情報と画像パスの取得
    # 最新の stats ファイルを探す簡易ロジック (run_Xフォルダ対応)
    import glob
    mae_txt = max(glob.glob(os.path.join(MAE_RESULTS_DIR, "**", f"global_best_{TARGET_ATTRIBUTE}_stats_full.txt"), recursive=True), key=os.path.getmtime)
    moco_txt = max(glob.glob(os.path.join(MOCO_RESULTS_DIR, "**", f"global_best_{TARGET_ATTRIBUTE}_stats_full.txt"), recursive=True), key=os.path.getmtime)
    
    mae_layer, mae_unit = get_sae_info(mae_txt, 'MAE')
    moco_layer, moco_unit = get_sae_info(moco_txt, 'MoCo')

    # 画像パスはMAE分析の結果を使用（同じ画像で比較するため）
    paths_txt = os.path.join(os.path.dirname(mae_txt), f"top_images_paths_{TARGET_ATTRIBUTE}.txt")
    if not os.path.exists(paths_txt):
        print(f"[ERROR] Paths file not found: {paths_txt}"); return
    with open(paths_txt, 'r') as f:
        image_paths = [l.strip() for l in f.readlines() if l.strip()][:9]

    if None in [mae_layer, mae_unit, moco_layer, moco_unit] or not image_paths:
        print("[ERROR] Failed to retrieve necessary information."); return

    print(f"  MAE Best SAE: L{mae_layer} U{mae_unit}")
    print(f"  MoCo Best SAE: L{moco_layer} U{moco_unit}")
    print(f"  Loaded {len(image_paths)} image paths.")

    # 2. モデル準備
    # MAE & SAE
    mae_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE).eval()
    sae_mae = SparseAutoencoder(cfg_mae.D_MODEL, cfg_mae.D_SAE, 0.0).to(DEVICE)
    sae_mae.load_state_dict(torch.load(cfg_mae.SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=mae_layer), map_location=DEVICE))
    sae_mae.eval()
    
    # MoCo & SAE
    moco_model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
    moco_model = load_moco_weights(moco_model).eval()
    sae_moco = SparseAutoencoder(cfg_moco.D_MODEL, cfg_moco.D_SAE, 0.0).to(DEVICE)
    sae_moco.load_state_dict(torch.load(cfg_moco.SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=moco_layer), map_location=DEVICE))
    sae_moco.eval()

    # 3. 描画ループ (3行6列: 左3列MAE, 右3列MoCo)
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i, path in enumerate(image_paths):
        row = i // 3
        col_mae = i % 3
        col_moco = col_mae + 3 # MoCoは右側の3列に配置

        img_pil = Image.open(path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        img_np = np.array(img_pil.resize((224, 224))) / 255.0

        # MAE Heatmap
        hm_mae, score_mae = create_heatmap(mae_model, sae_mae, img_tensor, mae_layer, mae_unit, 'mae_fc2')
        ax_mae = axes[row, col_mae]
        ax_mae.imshow(img_np)
        ax_mae.imshow(hm_mae, cmap='jet', alpha=0.5)
        ax_mae.set_title(f"MAE L{mae_layer} U{mae_unit}\nAct: {score_mae:.2f}", fontsize=9)
        ax_mae.axis('off')

        # MoCo Heatmap
        hm_moco, score_moco = create_heatmap(moco_model, sae_moco, img_tensor, moco_layer, moco_unit, 'moco_fc2')
        ax_moco = axes[row, col_moco]
        ax_moco.imshow(img_np)
        ax_moco.imshow(hm_moco, cmap='jet', alpha=0.5)
        ax_moco.set_title(f"MoCo L{moco_layer} U{moco_unit}\nAct: {score_moco:.2f}", fontsize=9)
        ax_moco.axis('off')

    # 全体タイトルと区切り線
    plt.suptitle(f"SAE Feature Visualization Comparison: '{TARGET_ATTRIBUTE}'\n(Left: MAE Top-9, Right: MoCo Top-9 on same images)", fontsize=16)
    # 中央に区切り線を入れる
    line = plt.Line2D([0.5, 0.5], [0.05, 0.92], transform=fig.transFigure, color="black", linewidth=2)
    fig.add_artist(line)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # タイトル分のスペースを空ける
    save_path = os.path.join(COMPARISON_SAVE_DIR, f"comparison_heatmap_{TARGET_ATTRIBUTE}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"=== Comparison heatmap saved to: {save_path} ===")

if __name__ == "__main__":
    main()