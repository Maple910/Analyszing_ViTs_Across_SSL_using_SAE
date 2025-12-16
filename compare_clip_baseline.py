# compare_clip_baseline
# CLIP 可視化

import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from torch.utils.data import Dataset, DataLoader

# 設定のインポート
from config_oid import OID_TRAIN_DIR, TARGET_ATTRIBUTE, CLIP_ANALYSIS_PATH, DEVICE

# --- 設定 ---
CLIP_MODEL_NAME = "ViT-B/16" 
BATCH_SIZE = 128

# データパス
OID_LABELS_CSV = os.path.join(OID_TRAIN_DIR, "labels.csv")
OID_IMAGES_DIR = os.path.join(OID_TRAIN_DIR, "dataset_images")

class OIDClipDataset(Dataset):
    def __init__(self, csv_path, img_dir, preprocess):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        img_path = os.path.join(self.img_dir, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
            return image, filename
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return torch.zeros(3, 224, 224), filename

def run_clip_analysis(target_class, top_k=9):
    print(f"--- Starting CLIP Zero-shot Analysis for '{target_class}' ---")
    
    # 1. モデルロード
    print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    model.eval()
    
    # 2. テキストエンベディング
    text_prompt = f"A photo of a {target_class}"
    text_tokens = clip.tokenize([text_prompt]).to(DEVICE)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    print(f"Text Prompt: '{text_prompt}' encoded.")

    # 3. データセット準備
    if not os.path.exists(OID_LABELS_CSV):
        print("Error: labels.csv not found.")
        return

    dataset = OIDClipDataset(OID_LABELS_CSV, OID_IMAGES_DIR, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    full_labels_df = pd.read_csv(OID_LABELS_CSV, index_col="filename")

    # 4. スコア計算
    print("Scanning OID dataset...")
    all_scores = []
    all_filenames = []
    
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Encoding Images"):
            images = images.to(DEVICE)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).squeeze()
            
            all_scores.append(similarity.cpu())
            all_filenames.extend(filenames)
            
    all_scores = torch.cat(all_scores)
    
    # 5. Top-K 抽出
    values, indices = torch.topk(all_scores, k=top_k)
    
    print(f"\nTop-{top_k} results for '{target_class}':")
    
    top_files = []
    top_scores = []
    pos_count = 0
    
    for i in range(top_k):
        idx = indices[i].item()
        score = values[i].item()
        filename = all_filenames[idx]
        
        top_files.append(filename)
        top_scores.append(score)
        
        is_pos = False
        try:
            labels_str = str(full_labels_df.loc[filename, 'labels'])
            if target_class in labels_str:
                is_pos = True
                pos_count += 1
        except: pass
        
        print(f"  {i+1}. {filename} (Score: {score:.2f}) -> {'✅ POS' if is_pos else '❌ NEG'}")

    selectivity = (pos_count / top_k) * 100
    print(f"CLIP Selectivity: {selectivity:.1f}%")

    # 6. 可視化
    clip_viz_dir = os.path.join(CLIP_ANALYSIS_PATH, "clip_visualization")
    os.makedirs(clip_viz_dir, exist_ok=True)
    
    cols = 3
    rows = ceil(top_k / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3.5))
    axes = axes.flatten()
    
    for i in range(top_k):
        filename = top_files[i]
        score = top_scores[i]
        img_path = os.path.join(OID_IMAGES_DIR, filename)
        
        is_pos = False
        try:
            if target_class in str(full_labels_df.loc[filename, 'labels']):
                is_pos = True
        except: pass
        
        label_text = "POS" if is_pos else "NEG"
        color = "green" if is_pos else "red"
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            axes[i].imshow(img)
            axes[i].set_title(f"CLIP Zero-shot\n{label_text} | Score: {score:.2f}", 
                              fontsize=10, color=color, fontweight='bold')
            axes[i].axis('off')
        except Exception as e:
            print(f"Error visualizing {filename}: {e}")
            
    for j in range(top_k, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"CLIP Zero-shot Retrieval: '{target_class}'\nPrompt: '{text_prompt}'", fontsize=14)
    plt.tight_layout()
    
    img_save_path = os.path.join(clip_viz_dir, f"clip_baseline_{target_class}.png")
    plt.savefig(img_save_path)
    print(f"\nVisualization saved to: {img_save_path}")
    plt.close()

    # 7. テキストファイル保存
    txt_save_path = os.path.join(clip_viz_dir, f"clip_baseline_{target_class}.png")
    
    with open(txt_save_path, 'w') as f:
        f.write(f"=== CLIP Zero-shot Analysis for Attribute: {target_class} ===\n\n")
        f.write(f"Model: {CLIP_MODEL_NAME}\n")
        f.write(f"Prompt: '{text_prompt}'\n")
        f.write(f"Dataset Size: {len(all_filenames)} images (Scanned from OID training subset)\n")
        f.write(f"Top-{top_k} Selectivity: {selectivity:.1f}% ({pos_count}/{top_k} are Positive)\n\n")
        
        f.write(f"--- Top-{top_k} Retrieval Results ---\n")
        for i in range(top_k):
            filename = top_files[i]
            score = top_scores[i]
            
            is_pos = False
            try:
                if target_class in str(full_labels_df.loc[filename, 'labels']):
                    is_pos = True
            except: pass
            
            label_str = "POS" if is_pos else "NEG"
            f.write(f"Rank {i+1}: {filename} (Score: {score:.4f}) -> {label_str}\n")
            
    print(f"Stats saved to: {txt_save_path}")

if __name__ == "__main__":
    run_clip_analysis(TARGET_ATTRIBUTE, top_k=9)