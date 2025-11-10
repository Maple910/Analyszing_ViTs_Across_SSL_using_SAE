# data_loader_celeba.py
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import config_celeba

# 再現性のためのシード設定
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CelebAAttributeDataset(Dataset):
    def __init__(self, img_dir, attr_path, transform=None, target_attribute='Blond_Hair', target_value=1, sample_size=None, random_seed=42):
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. 属性ファイルの読み込み
        # ファイル名をインデックスとして使用
        df = pd.read_csv(attr_path, sep='\s+', skiprows=1)
        
        if target_attribute not in df.columns:
            raise ValueError(f"Attribute '{target_attribute}' not found.")

        # 2. フィルタリング (例: Blond_Hair == 1 または Blond_Hair == -1)
        filtered_df = df[df[target_attribute] == target_value]

        # 3. サンプリングによるサイズ調整 (データセットのバランスを取るため)
        if sample_size is not None and len(filtered_df) > sample_size:
            random.seed(random_seed)
            filtered_df = filtered_df.sample(n=sample_size, random_state=random_seed)
        
        self.image_paths = [os.path.join(self.img_dir, filename) for filename in filtered_df.index]
        
        if not self.image_paths:
            raise RuntimeError(f"No images found for attribute '{target_attribute}' with value {target_value}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 特徴分析ではラベルは不要なのでダミーの0を返す
        label = 0 

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_celeba_attribute_loaders(img_dir, attr_path, batch_size, random_seed, num_images_to_sample):
    set_seed(random_seed) 
    
    # MAE/ViTモデルに適した標準的な前処理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 1. 属性あり (Positive: value = 1) データローダー
    dataset_attr = CelebAAttributeDataset(
        img_dir=img_dir, 
        attr_path=attr_path, 
        transform=transform, 
        target_attribute=config_celeba.TARGET_ATTRIBUTE, 
        target_value=1,
        sample_size=num_images_to_sample,
        random_seed=random_seed
    )

    # 2. 属性なし (Negative: value = -1) データローダー
    dataset_non_attr = CelebAAttributeDataset(
        img_dir=img_dir, 
        attr_path=attr_path, 
        transform=transform, 
        target_attribute=config_celeba.TARGET_ATTRIBUTE, 
        target_value=-1,
        sample_size=num_images_to_sample,
        random_seed=random_seed
    )

    # データセットサイズを確認し、一致しているか警告
    if len(dataset_attr) != len(dataset_non_attr):
        print(f"Warning: Dataset sizes are unequal ({config_celeba.TARGET_ATTRIBUTE} True: {len(dataset_attr)}, False: {len(dataset_non_attr)}).")
        
    dataloader_attr = DataLoader(dataset_attr, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_non_attr = DataLoader(dataset_non_attr, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataloader_attr, dataloader_non_attr