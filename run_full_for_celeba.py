import os
import subprocess
import time

# ==========================================
# ★実験したい属性リスト
# ==========================================
TARGET_ATTRIBUTES = [
    "Smiling",
    "Eyeglasses",
    "Male",
    "No_Beard",
    "Wavy_Hair",
    "Mouth_Slightly_Open",
    "Wearing_Lipstick",
    "Wearing_Hat",
    "Double_Chin",
    "Gray_Hair",
    "Mustache",
    "Bald"
]

# 設定ファイルのパス
CONFIG_PATH = "config_celeba.py"  # ★あなたの環境のファイル名に合わせてください

# 実行する分析スクリプト名
SCRIPT_NAME = "compare_attribute_feature_global_celeba.py"

def update_config_file(file_path, new_attribute):
    """Configファイルの TARGET_ATTRIBUTE を書き換える"""
    if not os.path.exists(file_path):
        print(f"[ERROR] Config file not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    updated = False
    for line in lines:
        if line.strip().startswith("TARGET_ATTRIBUTE ="):
            new_lines.append(f'TARGET_ATTRIBUTE = "{new_attribute}"\n')
            updated = True
        else:
            new_lines.append(line)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    if updated:
        print(f"  -> Updated {file_path} to '{new_attribute}'")
        return True
    else:
        print(f"  [WARNING] Could not find TARGET_ATTRIBUTE in {file_path}.")
        return False

def main():
    print("=== STARTING CELEBA BATCH EXPERIMENT ===")
    
    for attr in TARGET_ATTRIBUTES:
        print(f"\n\n{'='*50}")
        print(f" >>> Processing Attribute: {attr}")
        print(f"{'='*50}")

        # 1. Config更新
        if not update_config_file(CONFIG_PATH, attr):
            print("  [SKIP] Skipping due to config error.")
            continue
        
        # 2. 分析実行
        print(f"  -> Running {SCRIPT_NAME}...")
        start_time = time.time()
        
        # サブプロセスで実行
        ret = subprocess.run(f"python {SCRIPT_NAME}", shell=True)
        
        elapsed = time.time() - start_time
        
        if ret.returncode == 0:
            print(f"  [DONE] Finished in {elapsed:.1f} sec.")
        else:
            print(f"  [ERROR] Analysis failed for {attr} (Code: {ret.returncode})")

    print("\n\n=== ALL TASKS COMPLETED! ===")

if __name__ == "__main__":
    main()