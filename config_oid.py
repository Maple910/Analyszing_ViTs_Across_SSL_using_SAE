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
    0:  BASE_L1_COEFF * 1.0,  # OK
    1:  BASE_L1_COEFF * 0.5,  # OK
    2:  BASE_L1_COEFF * 0.2,  # OK
    
    # ------------要修正--------------
    3:  BASE_L1_COEFF * 0.02,  # 0.02 -> 0.01 (MaxAct 1.5超えを目指す)
    4:  BASE_L1_COEFF * 0.05,  # 0.05 -> 0.01 (0.53は低すぎるため劇的に下げる)
    5:  BASE_L1_COEFF * 0.02,  # 現状維持 (1.33 なのでOK)
    6:  BASE_L1_COEFF * 0.02,  # 0.02 -> 0.01 (0.85 なのであと少し緩和)
    7:  BASE_L1_COEFF * 0.01,  # 現状維持 (成功モデル)
    8:  BASE_L1_COEFF * 0.02,  # 0.02 -> 0.01 (0.88 なのであと少し緩和)
    # ------------要修正--------------
    
    9:  BASE_L1_COEFF * 0.2, # OK
    10: BASE_L1_COEFF * 0.2, # OK
    11: BASE_L1_COEFF * 0.5, # OK
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
# pos/negの場合は「_」でつなげるとちゃんとダウンロードしてくれないので半角スペースで
TARGET_ATTRIBUTE = "Microphone"

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
    "Animal",         # 【鳥】
    "Building",       # 【建築】
    "Furniture",      # 【椅子】
    "Clothing",       # 【布地】
    "Mobile phone",   # 【花】
    "Tree",           # 【風景】
]

# 最終的な訓練データパス
OID_TRAIN_DIR = os.path.join(OID_BASE_DIR, OID_TRAIN_DATASET_NAME)

# --- SAE重みのパス ---
# 汎用的なSAEとして保存 (属性名を含めない)
SAE_WEIGHTS_DIR = "./data/sae_weights_oid/for_dense_train_50k_each_2_run_11"
SAE_WEIGHTS_PATH_TEMPLATE = os.path.join(SAE_WEIGHTS_DIR, "sae_layer_{layer_idx}.pth")

# --- 結果の保存先 ---
# 分析結果のディレクトリ名に TARGET_ATTRIBUTE を使用
ANALYSIS_SAVE_DIR = f"./data/analysis_oid/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11"
ANALYSIS_PATH = os.path.join(".", ANALYSIS_SAVE_DIR)

NORMALIZE_ANALYSIS_DIR = f"./data/analysis_oid_normalize/analysis_results_oid_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_11"
NORMALIZE_PATH = os.path.join(".", NORMALIZE_ANALYSIS_DIR)

CLIP_ANALYSIS_SAVE_DIR =  f"./data/analysis_oid/analysis_results_oid_{TARGET_ATTRIBUTE}/"
CLIP_ANALYSIS_PATH = os.path.join(".", CLIP_ANALYSIS_SAVE_DIR)

# --- Weights & Biases ---
WANDB_PROJECT_NAME = "MAE_SAE_OpenImages"
WANDB_ENTITY = "hb220828-university-of-fukui"