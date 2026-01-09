# train_sae_moco.py
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

# ★変更: config_moco をインポート
from config_moco import *
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def train_sae_moco():
    set_seed(RANDOM_SEED)

    # 1. データローダーの準備
    train_images_path = os.path.join(OID_TRAIN_DIR, "dataset_images")
    print(f"Loading training data from: {train_images_path}")
    
    if not os.path.exists(train_images_path):
        print(f"Error: Training images directory not found at {train_images_path}")
        sys.exit(1)

    dataloader = get_oid_loader(train_images_path, BATCH_SIZE, shuffle=True, num_workers=4)
    data_loader_size = len(dataloader)
    print(f"Data loaded. Steps per epoch: {data_loader_size}")

    # 2. 保存ディレクトリの準備
    if os.path.exists(SAE_WEIGHTS_DIR):
        print(f"Error: Directory '{SAE_WEIGHTS_DIR}' already exists.")
        print("Please change 'SAE_WEIGHTS_DIR' in config_moco.py or remove the existing directory.")
        sys.exit(1)
        
    os.makedirs(SAE_WEIGHTS_DIR, exist_ok=True)
    
    # 3. モデルの準備 (★MoCo v3 ロード部分)
    print("Loading MoCo v3 Pre-trained ViT...")
    # まず普通のViTを作成
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=False).to(DEVICE)
    
    # 公式の重みをダウンロード
    url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location=DEVICE)
    
    # チェックポイントのキー構造を確認してロード
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # キーがない場合はcheckpointそのものがstate_dictの可能性あり
        state_dict = checkpoint

    # 重みのキーを整形 (module.base_encoder. を削除)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module.base_encoder."):
            new_key = k.replace("module.base_encoder.", "")
            new_state_dict[new_key] = v
            
    # ロード実行 (head周りの不一致は無視してOKなので strict=False)
    msg = vit_model.load_state_dict(new_state_dict, strict=False)
    print(f"MoCo weights loaded: {msg}")
    
    vit_model.eval()
    
    # 4. WandBの初期化
    wandb.init(
        project=WANDB_PROJECT_NAME, 
        entity=WANDB_ENTITY, 
        name=f"SAE_MoCo_Train_{SAE_WEIGHTS_DIR}", 
        config={
            "model": "MoCo_v3_ViT_Base", # モデル名を明記
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
        
        print(f"\nTraining SAE for layer {layer_idx}...")
        print(f"Applying L1 Coefficient: {current_l1_coeff:.2e}")
        
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, l1_coeff=current_l1_coeff).to(DEVICE)
        optimizer = torch.optim.Adam(sae_model.parameters(), lr=LEARNING_RATE)
        
        scaler = amp.GradScaler("cuda")
        
        for epoch in range(EPOCHS):
            total_loss = 0
            with tqdm(dataloader, desc=f"Layer {layer_idx} Ep {epoch+1}/{EPOCHS}") as pbar:
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
                    
                    with amp.autocast("cuda"):
                        reconstruction, sae_features = sae_model(layer_output)
                        reconstruction_loss = F.mse_loss(reconstruction, layer_output)
                        l1_loss = current_l1_coeff * torch.sum(torch.abs(sae_features)) / sae_features.shape[0] 
                        
                        sae_features_avg = sae_features.mean(dim=0)
                        ghost_grad_loss = GHOST_GRAD_COEFF * (sae_features_avg < 1e-6).sum()
                        
                        total_loss_val = reconstruction_loss + l1_loss + ghost_grad_loss

                    optimizer.zero_grad()
                    scaler.scale(total_loss_val).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += total_loss_val.item()
                    l0 = (sae_features > 0).float().sum(dim=1).mean().item()
                    
                    with torch.no_grad():
                        max_act_val = sae_features.max().item()

                    pbar.set_postfix(
                        loss=total_loss/(pbar.n+1), 
                        recon=reconstruction_loss.item(), 
                        l0=l0,
                        max=max_act_val
                    )

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
        
        sae_path = os.path.join(SAE_WEIGHTS_DIR, f"sae_layer_{layer_idx}.pth")
        torch.save(sae_model.state_dict(), sae_path)
        print(f"--> Saved: {sae_path}")

    wandb.finish()

if __name__ == "__main__":
    train_sae_moco()