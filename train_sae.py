# train_sae.py

# ... 既存のimport ...
import torch
import timm
import torch.nn.functional as F
from tqdm import tqdm
from sae_model import SparseAutoencoder
from data_loader import get_imagenet_val_dataloader
from config import *
import os
import sys
import wandb # wandbをインポート

# Function to register a forward hook to get the output of a specific layer
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def train_sae():
    # ディレクトリの存在チェック
    if os.path.exists(SAE_WEIGHTS_PATH):
        print(f"Error: Directory '{SAE_WEIGHTS_PATH}' already exists.")
        print("Please change 'SAE_TRAIN_DIR' in config.py or remove the existing directory.")
        sys.exit(1)
        
    os.makedirs(SAE_WEIGHTS_PATH, exist_ok=True)
    
    vit_model = timm.create_model(MAE_MODEL_NAME, pretrained=True).to(DEVICE)
    vit_model.eval()

    dataloader = get_imagenet_val_dataloader(DATASET_PATH, BATCH_SIZE, RANDOM_SEED)
    data_loader_size = len(dataloader) # <-- 追加: データローダーのサイズを取得
    
    # wandbを初期化
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config={
        "mae_model": MAE_MODEL_NAME,
        "d_model": D_MODEL,
        "d_sae": D_SAE,
        "l1_coeff": L1_COEFF,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "dataset_path": DATASET_PATH,
        "random_seed": RANDOM_SEED,
    })

    for layer_idx in range(12):
        print(f"Training SAE for layer {layer_idx}...")
        
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
        optimizer = torch.optim.Adam(sae_model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            total_loss = 0
            with tqdm(dataloader, desc=f"Layer {layer_idx} Epoch {epoch+1}") as pbar:
                for i, (images, _) in enumerate(pbar):
                    images = images.to(DEVICE)
                    
                    activations = {}
                    hook_handle = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(
                        get_activation(f"layer_{layer_idx}", activations)
                    )
                    
                    with torch.no_grad():
                        vit_model(images)
                        
                    hook_handle.remove()
                    
                    layer_output = activations[f"layer_{layer_idx}"].view(-1, D_MODEL)
                    
                    reconstruction, sae_features = sae_model(layer_output)
                    
                    reconstruction_loss = F.mse_loss(reconstruction, layer_output)
                    l1_loss = L1_COEFF * torch.sum(sae_features)
                    
                    sae_features_avg = sae_features.mean(dim=0)
                    ghost_grad_loss = GHOST_GRAD_COEFF * (sae_features_avg < 1e-6).sum()
                    total_loss_with_ghost_grads = reconstruction_loss + l1_loss + ghost_grad_loss
                    
                    optimizer.zero_grad()
                    total_loss_with_ghost_grads.backward()
                    optimizer.step()
                    
                    total_loss += total_loss_with_ghost_grads.item()
                    pbar.set_postfix(
                        loss=total_loss/(pbar.n+1), 
                        recon_loss=reconstruction_loss.item(), 
                        l1_loss=l1_loss.item(), 
                        ghost_grad=ghost_grad_loss.item()
                    )

                    # wandbにログを記録
                    
                    # ユーザーの要求に応じたカスタムX軸の計算
                    # レイヤー内での相対ステップ数 (Layer 0, 1, ...の開始時に0にリセットされる)
                    step_in_layer = i + epoch * data_loader_size

                    wandb.log({                      
                        "layer": layer_idx,
                        "epoch": epoch,
                        "total_loss": total_loss_with_ghost_grads.item(),
                        "reconstruction_loss": reconstruction_loss.item(),
                        "l1_loss": l1_loss.item(),
                        "ghost_grad_loss": ghost_grad_loss.item(),
                        "l0_norm": (sae_features > 0).float().sum(dim=1).mean().item(),
                        "mean_activation": sae_features.mean().item(),
                        
                        # 動的キーの追加
                        f"layer_{layer_idx}_total_loss": total_loss_with_ghost_grads.item(),
                        f"layer_{layer_idx}_reconstruction_loss": reconstruction_loss.item(), 
                        f"layer_{layer_idx}_l1_loss": l1_loss.item(), 
                        f"layer_{layer_idx}_ghost_grad_loss": ghost_grad_loss.item(), 
                        f"layer_{layer_idx}_l0_norm": (sae_features > 0).float().sum(dim=1).mean().item(), 
                        f"layer_{layer_idx}_step": step_in_layer, 
                    })
        
        sae_path = os.path.join(SAE_WEIGHTS_PATH, f"sae_layer_{layer_idx}.pth")
        torch.save(sae_model.state_dict(), sae_path)
        print(f"SAE model for layer {layer_idx} saved to {sae_path}")

    # wandbの実行を終了
    wandb.finish()

if __name__ == "__main__":
    train_sae()