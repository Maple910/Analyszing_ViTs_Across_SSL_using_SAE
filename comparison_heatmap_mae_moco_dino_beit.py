import os
import re
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms
import glob

# SAEモデル定義
from sae_model import SparseAutoencoder

# 各モデルの設定ファイルをインポート（パスや設定が異なる場合はここを調整してください）
import config_oid as cfg_mae      # MAE
import config_moco as cfg_moco    # MoCo
import config_beit as cfg_beit    # BEiT
import config_dino as cfg_dino    # DINO

# ==========================================
# ★設定項目（環境に合わせて書き換えてください）
# ==========================================
TARGET_ATTRIBUTE = "Mobile_phone" # 比較したい属性名

# 各モデルの分析結果（stats_full.txt）があるディレクトリ
MAE_DIR   = f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11"
MOCO_DIR  = f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"
BEIT_DIR  = f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"
DINO_DIR  = f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"

# 比較画像の保存先
SAVE_DIR = "./data/analysis_comparison_heatmaps/mae_moco_dino_beit/mae11_moco1_beit1_dino1"
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_sae_info(dir_path, attr, model_type):
    """ディレクトリから最新の stats ファイルを探し、ベストLayerとUnitを抽出"""
    pattern_file = os.path.join(dir_path, "**", f"global_best_{attr}_stats_full.txt")
    files = glob.glob(pattern_file, recursive=True)
    if not files:
        print(f"[ERROR] Stats file not found for {model_type} at {dir_path}")
        return None, None
    
    txt_path = max(files, key=os.path.getmtime)
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正規表現で Layer と Unit を抽出（表記の揺れに対応）
    patterns = [
        rf"GLOBAL BEST SAE(?: \({model_type}\))?: Layer (\d+), Unit (\d+)",
        r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)"
    ]
    
    for p in patterns:
        match = re.search(p, content)
        if match:
            return int(match.group(1)), int(match.group(2))
    
    print(f"[ERROR] Could not parse best unit for {model_type} in {txt_path}")
    return None, None

def load_moco_v3_weights(model):
    """MoCo v3の公式重みをロード"""
    url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
    sd = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    nsd = {k.replace("module.base_encoder.", ""): v for k, v in sd.items() if k.startswith("module.base_encoder.")}
    model.load_state_dict(nsd, strict=False)
    return model

def create_heatmap(model, sae, img_tensor, layer, unit_id):
    """指定されたモデルとSAEを使ってヒートマップを作成"""
    activations = {}
    def hook(m, i, o): activations['act'] = o.detach()
    
    # MLPのfc2出力をフック
    handle = model.blocks[layer].mlp.fc2.register_forward_hook(hook)
    with torch.no_grad():
        model(img_tensor)
    handle.remove()
    
    # パッチ部分の抽出 (1, 196, D) ※CLSトークンを除外
    raw_feats = activations['act'][:, 1:, :] 
    B, N, D = raw_feats.shape
    _, sae_flat = sae(raw_feats.reshape(-1, D))
    sae_feats = sae_flat.reshape(B, N, -1)
    
    # 特定ユニットのマップ
    target_map = sae_feats[0, :, unit_id]
    score = target_map.max().item()
    hm = target_map.reshape(14, 14).detach().cpu().numpy()
    hm = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # 正規化 (0-1)
    hm = np.maximum(hm, 0)
    if hm.max() > 0: hm /= hm.max()
    return hm, score

def main():
    print(f"=== Starting 4-Model Heatmap Comparison for: '{TARGET_ATTRIBUTE}' ===")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 各モデルの情報取得
    model_configs = [
        ("MAE", MAE_DIR, cfg_mae),
        ("MoCo", MOCO_DIR, cfg_moco),
        ("BEiT", BEIT_DIR, cfg_beit),
        ("DINO", DINO_DIR, cfg_dino)
    ]
    
    infos = {}
    for name, path, _ in model_configs:
        l, u = get_sae_info(path, TARGET_ATTRIBUTE, name)
        if l is None: return
        infos[name] = {"layer": l, "unit": u}

    # 2. バックボーンとSAEの準備
    models = {}
    saes = {}
    
    print("Loading models and SAEs...")
    # MAE
    models["MAE"] = timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE).eval()
    # MoCo
    moco_backbone = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
    models["MoCo"] = load_moco_v3_weights(moco_backbone).eval()
    # BEiT
    models["BEiT"] = timm.create_model("beit_base_patch16_224", pretrained=True).to(DEVICE).eval()
    # DINO
    models["DINO"] = timm.create_model("vit_base_patch16_224.dino", pretrained=True).to(DEVICE).eval()

    for name, _, cfg in model_configs:
        l = infos[name]["layer"]
        sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
        sae_path = cfg.SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=l)
        sae.load_state_dict(torch.load(sae_path, map_location=DEVICE))
        saes[name] = sae.eval()

    # 3. 画像パスの取得（MAEの解析結果にあるパスリストを使用）
    # ※パスリストが見つからない場合は MAE_DIR 直下などを確認してください
    paths_txt = glob.glob(os.path.join(BEIT_DIR, "**", f"top_images_paths_{TARGET_ATTRIBUTE}.txt"), recursive=True)
    if not paths_txt:
        print("[ERROR] Paths file not found."); return
    with open(paths_txt[0], 'r') as f:
        image_paths = [l.strip() for l in f.readlines() if l.strip()][:9]

    # 4. 描画設定: 9枚の画像に対し、それぞれ 2x2 の比較を行う
    # 配置: [MAE][MoCo]
    #       [BEiT][DINO]
    fig = plt.figure(figsize=(20, 24))
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3x3 の「画像グループ」を作り、その各マスの中に 2x2 の「モデル比較」を入れる
    outer_grid = fig.add_gridspec(3, 3, wspace=0.3, hspace=0.3)

    for i, path in enumerate(image_paths):
        inner_grid = outer_grid[i].subgridspec(2, 2, wspace=0.05, hspace=0.2)
        img_pil = Image.open(path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        img_np = np.array(img_pil.resize((224, 224))) / 255.0

        for j, (name, _, _) in enumerate(model_configs):
            ax = fig.add_subplot(inner_grid[j // 2, j % 2])
            l, u = infos[name]["layer"], infos[name]["unit"]
            
            hm, score = create_heatmap(models[name], saes[name], img_tensor, l, u)
            
            ax.imshow(img_np)
            ax.imshow(hm, cmap='jet', alpha=0.5)
            ax.set_title(f"{name} L{l} U{u}\nAct: {score:.2f}", fontsize=8)
            ax.axis('off')

    plt.suptitle(f"SAE Feature Comparison (2x2 Layout) - Attribute: '{TARGET_ATTRIBUTE}'\n[Top-Left: MAE, Top-Right: MoCo, Bottom-Left: BEiT, Bottom-Right: DINO]", fontsize=20)
    
    save_path = os.path.join(SAVE_DIR, f"heatmap_comparison_4models_{TARGET_ATTRIBUTE}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Success] Comparison heatmap saved to: {save_path}")

if __name__ == "__main__":
    main()