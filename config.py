# config.py
import torch
import os

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 再現性のためのシード値
RANDOM_SEED = 42

# モデル設定
MAE_MODEL_NAME = "vit_base_patch16_224"
D_MODEL = 768
D_SAE = D_MODEL * 32
L1_COEFF = 1e-9

# 訓練設定
PRETRAINED_MODEL_PATH = "mae_pretrain_model.pth"
DATASET_PATH = "./imagenet-val"
GHOST_GRAD_COEFF = 1e-4
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 1
LOG_INTERVAL = 10

# 可視化設定
NUM_IMAGES_TO_VISUALIZE = 16
LAYER_TO_ANALYZE = 1
FEATURE_TO_VISUALIZE = 14418
NEURON_TO_VISUALIZE = 500

# --- ディレクトリ構造の管理 ---

# ベースディレクトリ
BASE_DIR = "."

# 訓練時に生成されるSAE重みを保存するディレクトリ
# 新しい訓練を実行するたびに、このディレクトリ名を変えることで重みを隔離できます。
SAE_TRAIN_DIR = "sae_weights_run_7"
SAE_WEIGHTS_PATH = os.path.join(BASE_DIR, SAE_TRAIN_DIR)

# 可視化時に生成される画像を保存するディレクトリ
VISUALIZATION_SAVE_DIR = "visualizations"
VISUALIZATION_PATH = os.path.join(BASE_DIR, VISUALIZATION_SAVE_DIR)

# 可視化に使用するSAE重みのディレクトリ
# 複数のSAEを比較したい場合、ここにディレクトリパスを追加
SAE_VISUALIZE_DIRS = [
    "sae_weights_run_3"
]

# 可視化に使用するSAEの重みセットのインデックス
SAE_VISUALIZE_DIR_IDX = 0

# --- Weights & Biases (wandb) 設定 ---
WANDB_PROJECT_NAME = "MAE_SAE_Steering"
WANDB_ENTITY = "hb220828-university-of-fukui"