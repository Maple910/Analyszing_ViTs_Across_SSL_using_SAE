# 訓練

import torch
from torch import amp
import timm
import torch.nn.functional as F
from tqdm import tqdm
from sae_model import SparseAutoencoder 
from data_loader_celeba import get_celeba_attribute_loaders, set_seed, CelebAAttributeDataset 
from config_celeba import * 
import os
import sys
import wandb 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
import torchvision.transforms as transforms 

# DataLoaderに必要なDatasetの最小構成（すべてのCelebA画像を含む）
class FullCelebADataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.target_attribute = "Full CelebA"
        self.target_value = "All"
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Image, transforms をインポート
        image = Image.open(img_path).convert('RGB') 
        label = 0
        if self.transform:
            image = self.transform(image)
        return image, label

def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def train_sae_celeba():
    # CelebA全体を訓練に使うためのデータローダーを取得
    set_seed(RANDOM_SEED)

    # 全体のデータローダーを取得するための処理
    full_dataset = pd.read_csv(CELEBA_ATTR_PATH, delim_whitespace=True, skiprows=1)
    all_image_paths = [os.path.join(CELEBA_IMG_DIR, filename) for filename in full_dataset.index]
    
    # 前処理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset_instance = FullCelebADataset(all_image_paths, transform)
    dataloader = DataLoader(full_dataset_instance, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    data_loader_size = len(dataloader)

    # 訓練結果の保存ディレクトリを設定 (config_celeba.pyからSAE_WEIGHTS_DIRを使用)
    sae_weights_path = SAE_WEIGHTS_DIR
    if os.path.exists(sae_weights_path):
        print(f"Error: Directory '{sae_weights_path}' already exists.")
        print("Please change 'SAE_WEIGHTS_DIR' in config_celeba.py or remove the existing directory.")
        sys.exit(1)
        
    os.makedirs(sae_weights_path, exist_ok=True)
    
    vit_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True).to(DEVICE)
    vit_model.eval()
    
    # wandb初期化
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=f"SAE_CelebA_Train_{SAE_WEIGHTS_DIR}", config={
        "dataset": "CelebA",
        "d_model": D_MODEL,
        "d_sae": D_SAE,
        "l1_coeff": L1_COEFF,
        "ghost_grad_coeff": GHOST_GRAD_COEFF,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "random_seed": RANDOM_SEED,
    })

    for layer_idx in range(12):
        print(f"Training SAE for layer {layer_idx} on CelebA...")
        
        
        sae_model = SparseAutoencoder(D_MODEL, D_SAE, L1_COEFF).to(DEVICE)
        optimizer = torch.optim.Adam(sae_model.parameters(), lr=LEARNING_RATE)
        
        
        scaler = amp.GradScaler("cuda")
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
                    
                    # mixed precision（新 API: amp.autocast を使用）
                    with amp.autocast("cuda"):
                        reconstruction, sae_features = sae_model(layer_output)
                        reconstruction_loss = F.mse_loss(reconstruction, layer_output)
                        # L1 は絶対値で
                        l1_loss = L1_COEFF * torch.sum(torch.abs(sae_features))
                        sae_features_avg = sae_features.mean(dim=0)
                        ghost_grad_loss = GHOST_GRAD_COEFF * (sae_features_avg < 1e-6).sum()
                        total_loss_with_ghost_grads = reconstruction_loss + l1_loss + ghost_grad_loss

                    optimizer.zero_grad()
                    scaler.scale(total_loss_with_ghost_grads).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += total_loss_with_ghost_grads.item()
                    pbar.set_postfix(
                        loss=total_loss/(pbar.n+1), 
                        recon_loss=reconstruction_loss.item(), 
                        l1_loss=l1_loss.item(), 
                        ghost_grad=ghost_grad_loss.item()
                    )

                    # wandbにログを記録
                    step_in_layer = i + epoch * data_loader_size
                    
                    log_dict = {
                        "layer": layer_idx,
                        "epoch": epoch,
                        "total_loss": total_loss_with_ghost_grads.item(),
                        "reconstruction_loss": reconstruction_loss.item(),
                        "l1_loss": l1_loss.item(),
                        "ghost_grad_loss": ghost_grad_loss.item(),
                        "l0_norm": (sae_features > 0).float().sum(dim=1).mean().item(),
                        "mean_activation": sae_features.mean().item(),
                        
                        f"layer_{layer_idx}_total_loss": total_loss_with_ghost_grads.item(),
                        f"layer_{layer_idx}_reconstruction_loss": reconstruction_loss.item(),
                        f"layer_{layer_idx}_l1_loss": l1_loss.item(),
                        f"layer_{layer_idx}_ghost_grad_loss": ghost_grad_loss.item(),
                        f"layer_{layer_idx}_l0_norm": (sae_features > 0).float().sum(dim=1).mean().item(),
                        f"layer_{layer_idx}_step": step_in_layer, 
                    }
                    wandb.log(log_dict)
        
        # SAE重みの保存
        sae_path = os.path.join(sae_weights_path, f"sae_layer_{layer_idx}.pth")
        torch.save(sae_model.state_dict(), sae_path)
        print(f"SAE model for layer {layer_idx} saved to {sae_path}")

    wandb.finish()

if __name__ == "__main__":
    train_sae_celeba()