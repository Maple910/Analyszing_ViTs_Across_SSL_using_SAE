# train_sae_beit.py
import torch
from torch import amp
import timm
import torch.nn.functional as F
from tqdm import tqdm
from sae_model import SparseAutoencoder
from data_loader_oid import get_oid_loader, set_seed 
import os
import sys
import wandb

# BEiT 設定をインポート
from config_beit import *

def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def train_sae_beit():
    set_seed(RANDOM_SEED)

    # 1. データローダー
    train_images_path = os.path.join(OID_TRAIN_DIR, "dataset_images")
    if not os.path.exists(train_images_path):
        print(f"Error: Directory not found at {train_images_path}")
        sys.exit(1)
    dataloader = get_oid_loader(train_images_path, BATCH_SIZE, shuffle=True, num_workers=4)
    data_loader_size = len(dataloader)

    # 2. 保存先
    os.makedirs(SAE_WEIGHTS_DIR, exist_ok=True)
    
    # 3. モデルロード (timm経由でBEiTをロード)
    print(f"Loading Pre-trained BEiT: {MODEL_NAME}")
    vit_model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
    vit_model.eval()
    
    # 4. WandB初期化
    wandb.init(
        project=WANDB_PROJECT_NAME, 
        entity=WANDB_ENTITY, 
        name=f"SAE_BEiT_Train_{os.path.basename(SAE_WEIGHTS_DIR)}", 
        config={
            "model": MODEL_NAME,
            "dataset": OID_TRAIN_DATASET_NAME,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "base_l1_coeff": BASE_L1_COEFF,
            "l1_coeffs_dict": L1_COEFFS,
            "ghost_grad_coeff": GHOST_GRAD_COEFF,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "random_seed": RANDOM_SEED,
        }
    )

    # 5. 層ごとの訓練ループ
    for layer_idx in LAYERS_TO_ANALYZE: 
        current_l1_coeff = L1_COEFFS.get(layer_idx, BASE_L1_COEFF)
        print(f"\nTraining SAE for BEiT layer {layer_idx}...")
        
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, l1_coeff=current_l1_coeff).to(DEVICE)
        optimizer = torch.optim.Adam(sae_model.parameters(), lr=LEARNING_RATE)
        scaler = amp.GradScaler("cuda")
        
        for epoch in range(EPOCHS):
            total_loss = 0
            with tqdm(dataloader, desc=f"L{layer_idx} Ep {epoch+1}/{EPOCHS}") as pbar:
                for i, (images, _) in enumerate(pbar):
                    images = images.to(DEVICE)
                    activations = {}
                    hook = vit_model.blocks[layer_idx].mlp.fc2.register_forward_hook(get_activation("act", activations))
                    with torch.no_grad():
                        vit_model(images)
                    hook.remove()
                    
                    layer_output = activations["act"].view(-1, D_MODEL)
                    
                    with amp.autocast("cuda"):
                        recon, sf = sae_model(layer_output)
                        reconstruction_loss = F.mse_loss(recon, layer_output)
                        l1_loss = current_l1_coeff * torch.sum(torch.abs(sf)) / sf.shape[0] 
                        
                        sae_features_avg = sf.mean(dim=0)
                        ghost_grad_loss = GHOST_GRAD_COEFF * (sae_features_avg < 1e-6).sum()
                        total_loss_val = reconstruction_loss + l1_loss + ghost_grad_loss

                    optimizer.zero_grad()
                    scaler.scale(total_loss_val).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += total_loss_val.item()
                    l0 = (sf > 0).float().sum(dim=1).mean().item()
                    max_act_val = sf.max().item()

                    pbar.set_postfix(loss=total_loss/(i+1), l0=l0, max=max_act_val)

                    # MoCo/MAE版と完全一致させた記録項目
                    step_in_layer = i + epoch * data_loader_size
                    wandb.log({
                        "layer": layer_idx,
                        "epoch": epoch,
                        "total_loss": total_loss_val.item(),
                        "reconstruction_loss": reconstruction_loss.item(),
                        "l1_loss": l1_loss.item(),
                        "l0_norm": l0,
                        f"layer_{layer_idx}_total_loss": total_loss_val.item(),
                        f"layer_{layer_idx}_l0_norm": l0,
                        f"layer_{layer_idx}_recon_loss": reconstruction_loss.item(),
                        f"layer_{layer_idx}_step": step_in_layer, 
                        "max_feature_act": max_act_val,       
                        "current_l1_coeff": current_l1_coeff, 
                        f"layer_{layer_idx}_max_act": max_act_val,       
                        f"layer_{layer_idx}_l1_coeff": current_l1_coeff, 
                    })
        
        sae_path = SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer_idx)
        torch.save(sae_model.state_dict(), sae_path)
        print(f"--> Saved: {sae_path}")

    wandb.finish()

if __name__ == "__main__":
    train_sae_beit()