# config_moco.py
import torch
import os

# --- デバイス設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- モデル設定 ---
D_MODEL = 768 # ViT Base
D_SAE = D_MODEL * 32
GHOST_GRAD_COEFF = 1e-4

# MAEで成功した係数設定をそのまま使用 (公平な比較のため)
BASE_L1_COEFF = 3e-3 + 5e-4

L1_COEFFS = {
    0:  BASE_L1_COEFF * 1.0,
    1:  BASE_L1_COEFF * 0.5,
    2:  BASE_L1_COEFF * 0.2,
    3:  BASE_L1_COEFF * 0.01,
    4:  BASE_L1_COEFF * 0.01,
    5:  BASE_L1_COEFF * 0.02,
    6:  BASE_L1_COEFF * 0.01,
    7:  BASE_L1_COEFF * 0.01,
    8:  BASE_L1_COEFF * 0.01,
    9:  BASE_L1_COEFF * 0.2,
    10: BASE_L1_COEFF * 0.2,
    11: BASE_L1_COEFF * 0.5,
}

# --- 訓練・分析設定 ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 2
LAYERS_TO_ANALYZE = list(range(12)) # 全層、または特定の層 (例: [8, 9, 10, 11])
K_TOP_UNITS = 5
NUM_IMAGES_TO_VISUALIZE = 9

# --- 分析ターゲット設定 ---
TARGET_ATTRIBUTE = "Microphone"

# --- データセットパス (MAEと同じものを使用) ---
OID_BASE_DIR = "./data/oid_dataset" 
OID_TRAIN_DATASET_NAME = "dense_train_50k_each_2" 
OID_TRAIN_DIR = os.path.join(OID_BASE_DIR, OID_TRAIN_DATASET_NAME)

# --- SAE重みのパス (MoCo用に変更) ---
SAE_WEIGHTS_DIR = "./data/sae_weights_moco/for_dense_train_50k_each_2_run_1"
SAE_WEIGHTS_PATH_TEMPLATE = os.path.join(SAE_WEIGHTS_DIR, "sae_layer_{layer_idx}.pth")

# --- 結果の保存先 (MoCo用に変更) ---
ANALYSIS_SAVE_DIR = f"./data/analysis_moco/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"
ANALYSIS_PATH = os.path.join(".", ANALYSIS_SAVE_DIR)

NORMALIZE_ANALYSIS_DIR = f"./data/analysis_moco_normalize/analysis_results_moco_{TARGET_ATTRIBUTE}/for_dense_train_50k_each_2_run_1"
NORMALIZE_PATH = os.path.join(".", NORMALIZE_ANALYSIS_DIR)

# --- Weights & Biases ---
WANDB_PROJECT_NAME = "MoCo_SAE_OpenImages" # プロジェクト名を変更
WANDB_ENTITY = "hb220828-university-of-fukui"