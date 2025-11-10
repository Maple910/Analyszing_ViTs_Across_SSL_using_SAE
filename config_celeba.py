import torch
import os

# --- デバイス設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- モデル設定 ---
D_MODEL = 768 # ViT Baseの次元
D_SAE = D_MODEL * 32
L1_COEFF = 1e-6

# --- 訓練設定 ---
GHOST_GRAD_COEFF = 1e-5 
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 2

# --- CelebA データセットパス ---
#          画像データ                           属性情報
# ./data/celeba/img_align_celeba, ./data/celeba/list_attr_celeba.txt
CELEBA_BASE_DIR = "./data/celeba"
CELEBA_IMG_DIR = os.path.join(CELEBA_BASE_DIR, "img_align_celeba")
CELEBA_ATTR_PATH = os.path.join(CELEBA_BASE_DIR, "list_attr_celeba.txt")

# --- 分析設定 ---
LAYER_TO_ANALYZE = 10
K_TOP_UNITS = 20
NUM_IMAGES_TO_SAMPLE = 5000 
NUM_IMAGES_TO_VISUALIZE = 9 # 可視化枚数

# --- 分析ターゲット設定 ---
TARGET_ATTRIBUTE = "Blond_Hair" # 分析したい属性名を設定 (例: "Smiling", "Male")

# --- SAE重みのパス ---
SAE_WEIGHTS_DIR = "./data/sae_weights_celeba_run_4" 
SAE_WEIGHTS_PATH_TEMPLATE = os.path.join(SAE_WEIGHTS_DIR, "sae_layer_{layer_idx}.pth")

# --- 結果の保存先 ---
ANALYSIS_SAVE_DIR = f"./data/analysis_results_{TARGET_ATTRIBUTE}/sae_weights_celeba_run_4"
ANALYSIS_PATH = os.path.join("./", ANALYSIS_SAVE_DIR)

# --- Weights & Biases (wandb) 設定 ---
WANDB_PROJECT_NAME = "MAE_SAE_Steering_CelebA" 
WANDB_ENTITY = "hb220828-university-of-fukui"