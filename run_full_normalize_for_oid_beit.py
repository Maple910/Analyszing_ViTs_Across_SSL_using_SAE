# 指定した特徴群（TRAGET_ATTRIBUTES）に対して可視化を行う（BEiT用）

import os, subprocess, re, torch, timm, numpy as np, cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from sae_model import SparseAutoencoder
import config_beit as cfg_beit

TARGET_ATTRIBUTES = [    
    "Person",
    "Car",
    "Guitar",
    "Table",
    "Mobile_phone",
    "Bird",
    "Sunglasses",
    "Tree",
    "Building",
    "Chair",
    "Microphone"
]

CONFIG_PATH = "config_beit.py"
SCRIPT_COMPARE = "compare_attribute_normalization_oid_beit.py"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def update_config_file(p, attr):
    if not os.path.exists(p): return
    with open(p, 'r', encoding='utf-8') as f: lines = f.readlines()
    with open(p, 'w', encoding='utf-8') as f:
        for l in lines: f.write(f'TARGET_ATTRIBUTE = "{attr}"\n' if l.strip().startswith("TARGET_ATTRIBUTE =") else l)

def get_best_unit(p):
    if not os.path.exists(p): return None, None
    with open(p, 'r') as f:
        m = re.search(r"GLOBAL BEST SAE: Layer (\d+), Unit (\d+)", f.read())
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def generate_heatmap(attr, layer, unit, path):
    p_txt = os.path.join(path, f"top_images_paths_{attr}.txt")
    if not os.path.exists(p_txt): return
    with open(p_txt, 'r') as f: img_pths = [l.strip() for l in f.readlines() if l.strip()]
    save_dir = os.path.join(path, "patch_heatmaps"); os.makedirs(save_dir, exist_ok=True)
    
    vit = timm.create_model(cfg_beit.MODEL_NAME, pretrained=True).to(DEVICE).eval()
    sae = SparseAutoencoder(768, 768 * 32, 0.0).to(DEVICE)
    sae.load_state_dict(torch.load(cfg_beit.SAE_WEIGHTS_PATH_TEMPLATE.format(layer_idx=layer), map_location=DEVICE))
    sae.eval(); tr = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10)); axes = axes.flatten()
    print(f"Generating heatmaps for {attr} (L{layer} U{unit})...")
    
    for i, p in enumerate(img_pths[:9]):
        try: raw = Image.open(p).convert('RGB')
        except: continue
        acts = {}
        # ★絶対安全策: torch.no_grad()とdetach()を徹底
        with torch.no_grad():
            img_t = tr(raw).unsqueeze(0).to(DEVICE)
            hook = vit.blocks[layer].mlp.fc2.register_forward_hook(lambda m, i, o: acts.update({"f": o.detach()}))
            vit(img_t)
            hook.remove()
            _, sf = sae(acts["f"][:, 1:, :].reshape(-1, 768))
            # ★修正箇所: .detach()を追加して RuntimeError を回避
            target_unit_act = sf.view(196, -1)[:, unit].reshape(14, 14).detach().cpu().numpy()
            hm = cv2.resize(target_unit_act, (224, 224), interpolation=cv2.INTER_CUBIC)
            
        ax = axes[i]; img_np = np.array(raw.resize((224, 224)))/255.0
        ax.imshow(img_np); ax.imshow(hm/hm.max() if hm.max()>0 else hm, cmap='jet', alpha=0.5); ax.axis('off')
        
    save_path = os.path.join(save_dir, f"beit_heatmap_L{layer}_U{unit}_{attr}.png")
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f" [Success] Saved heatmap to: {save_path}")

def main():
    for attr in TARGET_ATTRIBUTES:
        print(f"\n>>> Processing: {attr}")
        update_config_file(CONFIG_PATH, attr)
        subprocess.run(f"python {SCRIPT_COMPARE}", shell=True)
        import glob
        base = cfg_beit.NORMALIZE_ANALYSIS_DIR.replace(f"_{cfg_beit.TARGET_ATTRIBUTE}", f"_{attr}")
        files = glob.glob(os.path.join(base, "**", f"global_best_{attr}_stats_full.txt"), recursive=True)
        if files:
            txt = max(files, key=os.path.getmtime); layer, unit = get_best_unit(txt)
            if layer is not None: generate_heatmap(attr, layer, unit, os.path.dirname(txt))

if __name__ == "__main__": main()