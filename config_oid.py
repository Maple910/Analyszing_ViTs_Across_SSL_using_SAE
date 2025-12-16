# config_oid.py
import torch
import os

# --- デバイス設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- モデル設定 ---
D_MODEL = 768 # ViT Base
D_SAE = D_MODEL * 32
GHOST_GRAD_COEFF = 1e-4

BASE_L1_COEFF = 3e-3 + 5e-4

L1_COEFFS = {
    0:  BASE_L1_COEFF * 1.0,  # Layer 0: 正常 (MaxAct ~2.7) -> 維持
    1:  BASE_L1_COEFF * 0.5,  # Layer 1: 少し弱い -> 半分に
    2:  BASE_L1_COEFF * 0.2,  # Layer 2-10: 瀕死 (MaxAct ~0.3) -> 1/5 に下げる
    3:  BASE_L1_COEFF * 0.2,
    4:  BASE_L1_COEFF * 0.2,
    5:  BASE_L1_COEFF * 0.2,
    6:  BASE_L1_COEFF * 0.2,
    7:  BASE_L1_COEFF * 0.2,
    8:  BASE_L1_COEFF * 0.2,
    9:  BASE_L1_COEFF * 0.2,
    10: BASE_L1_COEFF * 0.2,
    11: BASE_L1_COEFF * 0.5,  # Layer 11: 入力が大きいので少し下げる程度でOK    
}

# --- 訓練・分析設定 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 2
LAYERS_TO_ANALYZE = list(range(12))
K_TOP_UNITS = 5
NUM_IMAGES_TO_VISUALIZE = 9

# --- 分析ターゲット設定 ---
# Open Imagesのクラス名をそのまま指定します
# ディレクトリ名やファイル保存名にもこの文字列が使われます
TARGET_ATTRIBUTE = "Animal" 

# --- データセットパス ---
OID_BASE_DIR = "./data/oid_dataset" 
# 訓練データのフォルダ名をここで指定します
# 例1: "train_random_200k" (ランダムサンプリング)
# 例2: "dense_train_50k_each" (クラス指定の高密度サンプリング)
OID_TRAIN_DATASET_NAME = "dense_train_50k_each_2" 

# --- SAE訓練用クラス設定 (追加) ---
# SAEに汎用的な視覚特徴を学習させるためのベースとなるクラス群
GENERIC_CLASSES = [
    "Person",         # 【人物】
    "Car",            # 【乗り物】
    "Plant",          # 【植物】
    "Food",           # 【食事】
    "Animal",           # 【鳥】
    "Building",       # 【建築】
    "Furniture",          # 【椅子】
    "Clothing",       # 【布地】
    "Mobile phone",         # 【花】
    "Tree",           # 【風景】
]

# 最終的な訓練データパス
OID_TRAIN_DIR = os.path.join(OID_BASE_DIR, OID_TRAIN_DATASET_NAME)

# --- SAE重みのパス ---
# 汎用的なSAEとして保存 (属性名を含めない)
SAE_WEIGHTS_DIR = "./data/sae_weights_oid/for_dense_train_50k_each_2_run_7"
SAE_WEIGHTS_PATH_TEMPLATE = os.path.join(SAE_WEIGHTS_DIR, "sae_layer_{layer_idx}.pth")

# --- 結果の保存先 ---
# 分析結果のディレクトリ名に TARGET_ATTRIBUTE を使用
ANALYSIS_SAVE_DIR = f"./data/analysis_oid/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_7"
ANALYSIS_PATH = os.path.join(".", ANALYSIS_SAVE_DIR)

CLIP_ANALYSIS_SAVE_DIR =  f"./data/analysis_oid/analysis_results_oid_{TARGET_ATTRIBUTE}/"
CLIP_ANALYSIS_PATH = os.path.join(".", CLIP_ANALYSIS_SAVE_DIR)

# --- Weights & Biases ---
WANDB_PROJECT_NAME = "MAE_SAE_OpenImages"
WANDB_ENTITY = "hb220828-university-of-fukui"