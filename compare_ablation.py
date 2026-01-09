import os
import re
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sae_model import SparseAutoencoder
from tqdm import tqdm

# ==========================================
# ★設定項目：各リソースの場所を指定してください
# ==========================================

# 1. 解析対象の属性名（レポートのタイトル等に使用）
TARGET_ATTRIBUTE = "Microphone"

# 2. 各モデルのパス設定
# stats_path   : "GLOBAL BEST SAE: Layer X, Unit Y" が記載された統計ファイル
# weights_dir  : SAEの重みファイル（sae_layer_0.pth 〜 sae_layer_11.pth）が格納されているディレクトリ
# image_list   : 解析対象とする画像パスが並んだテキストファイル

MAE_PATHS = {
    "stats_path":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_oid/for_dense_train_50k_each_2_run_11",
    "image_list":   f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

MOCO_PATHS = {
    "stats_path":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_moco/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

DINO_PATHS = {
    "stats_path":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_dino/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

BEIT_PATHS = {
    "stats_path":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/global_best_{TARGET_ATTRIBUTE}_stats_full.txt",
    "weights_dir":  f"./data/sae_weights_beit/for_dense_train_50k_each_2_run_1",
    "image_list":   f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1/top_images_paths_{TARGET_ATTRIBUTE}.txt"
}

# 3. 結果の保存先
SAVE_DIR = f"./data/analysis_comparison_ablations/mae11_moco1_dino1_beit1/{TARGET_ATTRIBUTE}"

# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_unit_from_file(stats_path):
    """統計ファイルから Layer と Unit ID をパースする"""
    if not os.path.exists(stats_path):
        return None, None
    with open(stats_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # フォーマット「GLOBAL BEST SAE: Layer X, Unit Y」を検索
    match = re.search(r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)", content)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_backbone(model_type):
    """各手法のバックボーンをロード"""
    print(f"Loading {model_type} backbone...")
    if model_type == "MAE":
        return timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE).eval()
    elif model_type == "DINO":
        return timm.create_model("vit_base_patch16_224.dino", pretrained=True).to(DEVICE).eval()
    elif model_type == "BEiT":
        return timm.create_model("beit_base_patch16_224", pretrained=True).to(DEVICE).eval()
    elif model_type == "MoCo":
        model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
        url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
        sd = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        nsd = {k.replace("module.base_encoder.", ""): v for k, v in sd.items() if k.startswith("module.base_encoder.")}
        model.load_state_dict(nsd, strict=False)
        return model.eval()

def run_ablation(model_name, paths_dict):
    """手動指定されたディレクトリと統計ファイルから自動で重みを選んで実行"""
    stats_p = paths_dict["stats_path"]
    weights_dir = paths_dict["weights_dir"]
    images_p = paths_dict["image_list"]

    # 1. ユニット情報の取得 (Layer番号を取得)
    layer, unit_id = parse_unit_from_file(stats_p)
    if layer is None:
        print(f" [Error] {model_name} の統計ファイルからユニット情報を読み取れませんでした: {stats_p}")
        return None, None
    
    # ★修正箇所: Layer番号に基づいてファイル名を生成
    weights_p = os.path.join(weights_dir, f"sae_layer_{layer}.pth")
    
    if not os.path.exists(weights_p):
        print(f" [Error] {model_name} の重みファイルが存在しません: {weights_p}")
        return None, None

    if not os.path.exists(images_p):
        print(f" [Error] {model_name} の画像リストファイルが存在しません: {images_p}")
        return None, None

    # 2. モデルとSAEのロード
    vit = load_backbone(model_name)
    sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
    sae.load_state_dict(torch.load(weights_p, map_location=DEVICE))
    sae.eval()

    # 画像パスの読み込み
    with open(images_p, 'r', encoding='utf-8') as f:
        img_paths = [l.strip() for l in f.readlines() if l.strip()][:9]

    tr = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    acts = {}
    handle = vit.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: acts.update({'f': o.detach()}))

    drops = []
    for p in img_paths:
        try:
            img_t = tr(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
        except:
            continue

        with torch.no_grad():
            # オリジナルの最大活性
            vit(img_t)
            _, f_all = sae(acts['f'][:, 1:, :].reshape(-1, 768)) 
            f_patch = f_all.view(196, -1)[:, unit_id]
            orig_max = f_patch.max().item()
            if orig_max < 1e-6: continue
            
            # 最大パッチ特定と消去
            max_idx = torch.argmax(f_patch).item()
            grid_y, grid_x = divmod(max_idx, 14)
            abl_t = img_t.clone()
            abl_t[:, :, grid_y*16:(grid_y+1)*16, grid_x*16:(grid_x+1)*16] = 0
            
            # 再推論
            vit(abl_t)
            _, f_abl_all = sae(acts['f'][:, 1:, :].reshape(-1, 768))
            abl_max = f_abl_all.view(196, -1)[:, unit_id].max().item()
            
            # 減少率計算
            drop = (orig_max - abl_max) / (orig_max + 1e-8) * 100
            drops.append(max(0, drop))
    
    handle.remove()
    avg_drop = np.mean(drops) if drops else None
    return avg_drop, (layer, unit_id)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    report_file = os.path.join(SAVE_DIR, f"ablation_report_{TARGET_ATTRIBUTE}.txt")

    tasks = [
        ("MAE",  MAE_PATHS),
        ("MoCo", MOCO_PATHS),
        ("DINO", DINO_PATHS),
        ("BEiT", BEIT_PATHS)
    ]

    print(f"\n{'='*60}\n SSL Model Ablation Analysis (Manual Dir Mode): {TARGET_ATTRIBUTE}\n{'='*60}")

    with open(report_file, 'w', encoding='utf-8') as f_report:
        f_report.write(f"=== SSL Model Activation Decay (Ablation) Report ===\n")
        f_report.write(f"Attribute: {TARGET_ATTRIBUTE}\n\n")
        f_report.write(f"{'Model':<10} | {'Layer':<5} | {'Unit':<7} | {'Avg Drop (%)':<15}\n")
        f_report.write("-" * 55 + "\n")

        for m_name, m_paths in tasks:
            print(f"\n>>> Processing {m_name}...")
            drop, info = run_ablation(m_name, m_paths)
            if drop is not None:
                print(f"  -> Best Layer: {info[0]}, Unit: {info[1]}")
                print(f"  -> Average Drop: {drop:.2f}%")
                f_report.write(f"{m_name:<10} | {info[0]:<5} | {info[1]:<7} | {drop:>10.2f}%\n")
            else:
                print(f"  -> [Skip] Failed to process {m_name}. Check the logs.")

    print(f"\n[Success] Report saved to: {report_file}")

if __name__ == "__main__":
    main()