import os
import subprocess
import re
import time
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# 必要なモジュールをインポート
from sae_model import SparseAutoencoder
from data_loader_oid import get_openimages_attribute_loaders

# 定数読み込み用 (パス構造などはここから借ります)
import config_moco as cfg_moco

# ==========================================
# ★実験したい属性リスト (ここを編集)
# ==========================================
TARGET_ATTRIBUTES = [
    #"Person",
    #"Car",
    #"Guitar",
    "Vehicle",
    "Table",
    "Chair",
    
    "Mobile_phone",
    "Bird",
    "Sunglasses",
    "Microphone",
    "Tree",
    "Furniture"
    # 必要なだけ追加してください
]
# ==========================================

# ファイルパス設定
CONFIG_PATH = "config_moco.py"
SCRIPT_COMPARE = "compare_attribute_feature_global_oid_moco.py"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def update_config_file(file_path, new_attribute):
    """Configファイルの TARGET_ATTRIBUTE を書き換える"""
    if not os.path.exists(file_path):
        print(f"[ERROR] Config file not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    updated = False
    for line in lines:
        if line.strip().startswith("TARGET_ATTRIBUTE ="):
            new_lines.append(f'TARGET_ATTRIBUTE = "{new_attribute}"\n')
            updated = True
        else:
            new_lines.append(line)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    if updated:
        print(f"  -> Updated {file_path} to '{new_attribute}'")
    else:
        print(f"  [WARNING] Could not update {file_path}.")

def get_best_unit_from_txt(txt_path):
    """分析結果テキストからベストなLayerとUnitを抜き出す"""
    if not os.path.exists(txt_path):
        print(f"  [ERROR] Stats file not found: {txt_path}")
        return None, None

    with open(txt_path, 'r') as f:
        content = f.read()

    # 正規表現で検索: "GLOBAL BEST SAE (MoCo): Layer X, Unit Y"
    pattern = r"GLOBAL BEST SAE \(MoCo\): Layer (\d+), Unit (\d+)"
    match = re.search(pattern, content)
    
    # 見つからない場合の予備検索 (フォーマット揺れ対応)
    if not match:
        pattern = r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)"
        match = re.search(pattern, content)

    if match:
        layer = int(match.group(1))
        unit = int(match.group(2))
        return layer, unit
    else:
        print(f"  [ERROR] Could not parse Best Unit from {txt_path}")
        return None, None

# --- MoCo専用 ヒートマップ生成関数（修正版） ---
def generate_moco_heatmap(attribute, layer, unit_id, analysis_path):
    print(f"  -> Generating Consistent MoCo Heatmap | {attribute} | L{layer} U{unit_id}...")
    
    # 分析コードが保存した「パスリスト」のファイルを特定
    paths_txt_path = os.path.join(analysis_path, f"top_images_paths_{attribute}.txt")
    if not os.path.exists(paths_txt_path):
        print(f"  [ERROR] Top image paths file not found: {paths_txt_path}")
        return

    with open(paths_txt_path, 'r') as f:
        target_image_paths = [line.strip() for line in f.readlines() if line.strip()]

    # 保存先ディレクトリ
    save_dir = os.path.join(analysis_path, "patch_heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. MoCoモデルロード
    print("  -> Loading MoCo v3 model...")
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
    
    # MoCo v3 重みのロード
    url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
    
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module.base_encoder."):
            new_key = k.replace("module.base_encoder.", "")
            new_state_dict[new_key] = v
    vit_model.load_state_dict(new_state_dict, strict=False)
    vit_model.eval()

    # 2. SAEロード
    sae_path = cfg_moco.SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer)
    if not os.path.exists(sae_path):
        print(f"  [SKIP] SAE weights not found: {sae_path}")
        return

    sae_model = SparseAutoencoder(cfg_moco.D_MODEL, cfg_moco.D_SAE, 0.0).to(DEVICE)
    sae_model.load_state_dict(torch.load(sae_path, map_location=DEVICE))
    sae_model.eval()

    # 可視化用の変換設定
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 描画の準備
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    activations = {}
    def get_act(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # 指定されたパスの画像（最大9枚）のみを処理
    for i, img_path in enumerate(target_image_paths[:9]):
        raw_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            hook = vit_model.blocks[layer].mlp.fc2.register_forward_hook(get_act("fc2"))
            vit_model(img_tensor)
            hook.remove()
            
            # SAE処理
            raw_feats = activations["fc2"][:, 1:, :] # (1, 196, 768)
            B, N, D = raw_feats.shape
            _, sae_flat = sae_model(raw_feats.reshape(-1, D))
            sae_feats = sae_flat.reshape(B, N, cfg_moco.D_SAE)
            
            # ヒートマップ抽出
            target_map = sae_feats[0, :, unit_id] # (196,)
            score = target_map.max().item()
            hm = target_map.reshape(14, 14).cpu().numpy()
            hm = cv2.resize(hm, (224, 224), interpolation=cv2.INTER_CUBIC)
            hm = np.maximum(hm, 0)
            if hm.max() > 0: hm /= hm.max()
            
            # 表示用のDenormalize画像作成
            img_np = np.array(raw_img.resize((224, 224))) / 255.0
            
            ax = axes[i]
            ax.imshow(img_np)
            ax.imshow(hm, cmap='jet', alpha=0.5)
            ax.set_title(f"Act: {score:.2f}", fontsize=9, fontweight='bold')
            ax.axis('off')

    plt.suptitle(f"MoCo | {attribute} | L{layer} Unit {unit_id}", fontsize=14)
    plt.tight_layout()
    
    save_name = f"moco_heatmap_consistent_L{layer}_U{unit_id}_{attribute}.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved consistent heatmap: {save_path}")


# ==========================================
# メインループ
# ==========================================
def main():
    print("=== STARTING MoCo BATCH EXPERIMENT ===")
    
    for attr in TARGET_ATTRIBUTES:
        print(f"\n\n{'='*50}")
        print(f" >>> Processing Attribute: {attr}")
        print(f"{'='*50}")

        # 1. Config更新
        update_config_file(CONFIG_PATH, attr)
        
        # 2. 分析実行 (subprocess)
        print("  -> Running MoCo Analysis Script...")
        # エラーが出ても止まらないように check=False 相当で実行
        ret = subprocess.run(f"python {SCRIPT_COMPARE}", shell=True)
        
        if ret.returncode != 0:
            print("  [ERROR] MoCo Analysis failed. Skipping viz.")
            continue

        # 3. 結果読み込み & ヒートマップ生成
        # 保存先パスの推定
        base_dir = cfg_moco.ANALYSIS_SAVE_DIR.replace(f"_{cfg_moco.TARGET_ATTRIBUTE}", f"_{attr}")
        txt_name = f"global_best_{attr}_stats_full.txt"
        
        # 実際の出力パス（run_X 等の階層が含まれる可能性があるため検索）
        import glob
        pattern = os.path.join(base_dir, "**", txt_name)
        potential_files = glob.glob(pattern, recursive=True)
        
        if not potential_files:
            print(f"  [ERROR] Could not find {txt_name} in {base_dir}")
            continue
            
        # 最も新しく作成されたファイルを対象とする
        txt_path = max(potential_files, key=os.path.getmtime)
        analysis_path = os.path.dirname(txt_path)
        
        layer, unit = get_best_unit_from_txt(txt_path)
        
        if layer is not None:
            # 修正した分析パスを渡してヒートマップ作成
            generate_moco_heatmap(attr, layer, unit, analysis_path)
        else:
            print(f"  [WARN] Could not determine best unit for {attr}.")

    print("\n\n=== ALL TASKS COMPLETED! ===")

if __name__ == "__main__":
    main()