# config_beit.py
import torch
import os

# --- デバイス設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- モデル設定 ---
# timm でサポートされている BEiT ViT-Base
MODEL_NAME = "beit_base_patch16_224" 
D_MODEL = 768 
D_SAE = D_MODEL * 32
GHOST_GRAD_COEFF = 1e-4

# BEiTの活性化スケールに最適化したL1係数
# 4e-4 前後が、デッドニューロンを抑えつつ解釈性を担保するのに適しています
BASE_L1_COEFF = 4e-4

L1_COEFFS = {
    0:  BASE_L1_COEFF * 1.0,
    1:  BASE_L1_COEFF * 1.0,
    2:  BASE_L1_COEFF * 1.0,
    3:  BASE_L1_COEFF * 1.0,
    4:  BASE_L1_COEFF * 1.0,
    5:  BASE_L1_COEFF * 1.0,
    6:  BASE_L1_COEFF * 1.0,
    7:  BASE_L1_COEFF * 1.0,
    8:  BASE_L1_COEFF * 0.8, # 深層は情報密度が高いため少し緩和
    9:  BASE_L1_COEFF * 0.7,
    10: BASE_L1_COEFF * 0.6,
    11: BASE_L1_COEFF * 0.5,
}

# --- 訓練・分析設定 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 2
LAYERS_TO_ANALYZE = list(range(12))
NUM_IMAGES_TO_VISUALIZE = 9

# --- 分析ターゲット設定 ---
TARGET_ATTRIBUTE = "Microphone"

# --- データセットパス ---
OID_BASE_DIR = "./data/oid_dataset" 
OID_TRAIN_DATASET_NAME = "dense_train_50k_each_2" 
OID_TRAIN_DIR = os.path.join(OID_BASE_DIR, OID_TRAIN_DATASET_NAME)

# --- SAE重みのパス (BEiT用) ---
SAE_WEIGHTS_DIR = "./data/sae_weights_beit/for_dense_train_50k_each_2_run_1"
SAE_WEIGHTS_PATH_TEMPLATE = os.path.join(SAE_WEIGHTS_DIR, "sae_layer_{layer_idx}.pth")

# --- 結果の保存先 (BEiT用) ---
NORMALIZE_ANALYSIS_DIR = f"./data/analysis_beit_normalize/analysis_results_beit_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"
NORMALIZE_PATH = os.path.join(".", NORMALIZE_ANALYSIS_DIR)

# --- Weights & Biases ---
# 他モデルと揃え、プロジェクト内で比較しやすくするための設定
WANDB_PROJECT_NAME = "BEiT_SAE_OpenImages"
WANDB_ENTITY = "hb220828-university-of-fukui"