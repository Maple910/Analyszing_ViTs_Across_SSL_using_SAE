# config_dino.py
import torch
import os

# --- デバイス設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- モデル設定 ---
# DINO v1 ViT-Base (timmサポート名)
MODEL_NAME = "vit_base_patch16_224.dino"
D_MODEL = 768 
D_SAE = D_MODEL * 32
GHOST_GRAD_COEFF = 1e-4

# MoCoの設定（3e-3 + 5e-4）と完全に一致させる（公平な比較のため）
BASE_L1_COEFF = 3e-3 + 5e-4

# MoCoの設定からそのまま引き継ぎ
L1_COEFFS = {
    0:  0.00008,
    1:  0.00008,
    2:  0.00008,
    3:  0.00008,
    4:  0.00008,
    5:  0.00008,
    6:  0.00008,
    7:  0.00008,
    8:  0.00007, # 中層から少しずつ調整
    9:  0.00006,
    10: 0.00005,
    11: 0.00004,
}

# --- 訓練・分析設定 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 2
LAYERS_TO_ANALYZE = list(range(12))
K_TOP_UNITS = 5
NUM_IMAGES_TO_VISUALIZE = 9

# --- 分析ターゲット設定 ---
TARGET_ATTRIBUTE = "Microphone"

# --- データセットパス ---
OID_BASE_DIR = "./data/oid_dataset" 
OID_TRAIN_DATASET_NAME = "dense_train_50k_each_2" 
OID_TRAIN_DIR = os.path.join(OID_BASE_DIR, OID_TRAIN_DATASET_NAME)

# --- SAE重みのパス (DINO用に変更) ---
SAE_WEIGHTS_DIR = "./data/sae_weights_dino/for_dense_train_50k_each_2_run_1"
SAE_WEIGHTS_PATH_TEMPLATE = os.path.join(SAE_WEIGHTS_DIR, "sae_layer_{layer_idx}.pth")

# --- 結果の保存先 (DINO用に変更) ---
ANALYSIS_SAVE_DIR = f"./data/analysis_dino/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"
ANALYSIS_PATH = os.path.join(".", ANALYSIS_SAVE_DIR)

NORMALIZE_ANALYSIS_DIR = f"./data/analysis_dino_normalize/analysis_results_dino_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"
NORMALIZE_PATH = os.path.join(".", NORMALIZE_ANALYSIS_DIR)

# --- Weights & Biases ---
WANDB_PROJECT_NAME = "DINO_SAE_OpenImages"
WANDB_ENTITY = "hb220828-university-of-fukui"