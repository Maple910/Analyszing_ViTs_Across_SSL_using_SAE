# data_loader_oid.py
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class OpenImagesFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 画像ファイルを取得
        self.image_paths = sorted(glob(os.path.join(root_dir, "*.jpg")))
        
        if not self.image_paths:
             raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            return torch.zeros((3, 224, 224)), 0

        label = 0 
        if self.transform:
            image = self.transform(image)
        return image, label

def get_oid_loader(root_dir, batch_size, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = OpenImagesFolderDataset(root_dir=root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# --- 分析用ラッパー ---

def get_openimages_attribute_loaders(base_dir, attr_name, batch_size, random_seed, num_images_to_sample=None):
    """
    attr_name には TARGET_ATTRIBUTE が渡される想定
    """
    set_seed(random_seed)
    
    # 属性別フォルダパス (./data/oid_dataset/Guitar/positive などを参照)
    attr_dir = os.path.join(base_dir, attr_name, "positive")
    non_attr_dir = os.path.join(base_dir, attr_name, "negative")
    
    loader_attr = get_oid_loader(attr_dir, batch_size, shuffle=False)
    loader_non_attr = get_oid_loader(non_attr_dir, batch_size, shuffle=False)
    
    return loader_attr, loader_non_attr