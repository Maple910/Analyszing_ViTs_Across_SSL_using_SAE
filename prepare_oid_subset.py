# prepare_oid_subset.py
# 訓練に用いるデータセット作成

import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil
from PIL import Image
from tqdm import tqdm
import random
import csv
from collections import defaultdict
from config_oid import OID_BASE_DIR, OID_TRAIN_DIR, RANDOM_SEED, GENERIC_CLASSES

# --- 訓練対象クラス ---
DENSE_TARGET_CLASSES = GENERIC_CLASSES

def resize_and_move_images(src_dir, dest_dir, size=256):
    """
    src_dirにある画像をリサイズしてdest_dirに移動する。
    
    Returns:
        tuple: (moved_count, batch_filenames)
        - moved_count: 新規に保存（移動）された画像の枚数
        - batch_filenames: このバッチに含まれていた全画像のファイル名リスト
    """
    if not os.path.exists(src_dir): return 0, []
    
    # dest_dir (dataset_images) がなければ作成
    os.makedirs(dest_dir, exist_ok=True)
    
    image_paths = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    batch_filenames = []
    moved_count = 0
    
    for filepath in tqdm(image_paths, desc="Resizing & Moving"):
        filename = os.path.basename(filepath)
        dest_path = os.path.join(dest_dir, filename)
        
        # このバッチ（クラス）に含まれる画像としてリストに追加
        batch_filenames.append(filename)
        
        # 既に存在する場合（＝他のクラスでダウンロード済み）
        # ファイル移動はスキップする
        if os.path.exists(dest_path):
            continue
            
        try:
            with Image.open(filepath) as img:
                img = img.convert('RGB')
                
                width, height = img.size
                if width < height:
                    new_width = size
                    new_height = int(height * (size / width))
                else:
                    new_height = size
                    new_width = int(width * (size / height))
                
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                left = (new_width - size) / 2
                top = (new_height - size) / 2
                right = (new_width + size) / 2
                bottom = (new_height + size) / 2
                img_cropped = img_resized.crop((left, top, right, bottom))
                
                img_cropped.save(dest_path, optimize=True, quality=85)
                moved_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # エラー時はリストから除外すべきだが、今回は簡易的に
            
    return moved_count, batch_filenames

def save_statistics(output_root_dir, class_counts, unique_files_count):
    """
    クラスごとの枚数内訳をテキストファイルに保存
    """
    stats_path = os.path.join(output_root_dir, "dataset_stats.txt")
    
    # 論理的な延べ枚数（重複込み）
    total_logical_count = sum(class_counts.values())
    
    with open(stats_path, "w") as f:
        f.write("=== OID Dense Subset Statistics ===\n")
        f.write(f"Unique Images on Disk: {unique_files_count}\n")
        f.write(f"Total Logical Samples: {total_logical_count} (Sum of class counts including overlaps)\n\n")
        f.write("--- Breakdown by Class (Logical Counts) ---\n")
        f.write("Note: 'Person' count includes images that are also 'Car', etc.\n\n")
        
        # 枚数が多い順にソートして表示
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cls, count in sorted_counts:
            f.write(f"{cls}: {count}\n")
            
    print(f"\nStatistics saved to: {stats_path}")

def save_labels_csv(output_root_dir, file_to_labels):
    """
    画像ごとのラベル情報をCSVファイルに保存 (CLIP評価用)
    Format: filename, labels (e.g. "Car,Person")
    """
    csv_path = os.path.join(output_root_dir, "labels.csv")
    print(f"\nWriting labels to {csv_path}...")
    
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "labels"]) # Header
            
            # ファイル名でソートして書き込み
            for filename in sorted(file_to_labels.keys()):
                # ラベルセットをアルファベット順にソートしてカンマ区切り文字列に
                labels = sorted(list(file_to_labels[filename]))
                label_str = ",".join(labels)
                writer.writerow([filename, label_str])
                
        print(f"Labels CSV saved successfully.")
        
    except Exception as e:
        print(f"Error saving labels csv: {e}")

def prepare_dense_oid_dataset(output_root_dir, samples_per_class=50000):
    cache_dir = os.path.join(os.path.dirname(OID_BASE_DIR), "fiftyone_cache")
    fo.config.dataset_zoo_dir = cache_dir
    
    images_dir = os.path.join(output_root_dir, "dataset_images")
    
    target_total = len(DENSE_TARGET_CLASSES) * samples_per_class
    
    print(f"--- Preparing High-Density Open Images Subset ---")
    print(f"Dataset Root : {output_root_dir}")
    print(f"Images Dir   : {images_dir}")
    print(f"Target Classes: {DENSE_TARGET_CLASSES}")
    
    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        print("Initializing metadata...")
        dummy = foz.load_zoo_dataset("open-images-v7", split="train", max_samples=1, shuffle=True, seed=RANDOM_SEED, dataset_name="oid_meta", drop_existing_dataset=True)
        dummy.delete()
    except Exception as e:
        print(f"Metadata init warning: {e}")

    # 統計用: クラスごとの「論理的な」枚数
    class_counts = defaultdict(int)
    # ラベル用: ファイル名 -> ラベルセットの辞書
    file_to_labels = defaultdict(set)
    
    for i, cls in enumerate(DENSE_TARGET_CLASSES):
        print(f"\n[{i+1}/{len(DENSE_TARGET_CLASSES)}] Processing class: '{cls}'")
        
        try:
            print(f"Downloading {samples_per_class} images...")
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],
                classes=[cls],
                max_samples=samples_per_class,
                shuffle=True,
                seed=RANDOM_SEED,
                dataset_name="temp_class_loader",
                drop_existing_dataset=True
            )
            
            temp_export_dir = os.path.join(cache_dir, "temp_export")
            if os.path.exists(temp_export_dir): shutil.rmtree(temp_export_dir)
            
            dataset.export(
                export_dir=temp_export_dir,
                dataset_type=fo.types.ImageDirectory,
            )
            
            # リサイズ移動処理 & ファイル名リストの取得
            saved_count, batch_filenames = resize_and_move_images(temp_export_dir, images_dir, size=256)
            
            # 統計情報の更新
            class_counts[cls] = len(batch_filenames)
            
            # ラベル情報の更新 (このクラスで取得された全画像にラベルを付与)
            for fname in batch_filenames:
                file_to_labels[fname].add(cls)
            
            print(f"-> Class '{cls}': {len(batch_filenames)} images found. (New files added: {saved_count})")
            
            dataset.delete()
            if os.path.exists(temp_export_dir): shutil.rmtree(temp_export_dir)
            
            source_images_dir = os.path.join(cache_dir, "open-images-v7", "train", "data")
            if os.path.exists(source_images_dir):
                shutil.rmtree(source_images_dir)
                os.makedirs(source_images_dir, exist_ok=True)

        except Exception as e:
            print(f"Error processing class {cls}: {e}")
            continue

    # 最終的なディスク上のユニーク枚数
    total_unique_files = len([name for name in os.listdir(images_dir) if name.endswith('.jpg')])
    
    # 統計情報の保存 (dataset_stats.txt)
    save_statistics(output_root_dir, class_counts, total_unique_files)
    
    # ラベルCSVの保存 (labels.csv)
    save_labels_csv(output_root_dir, file_to_labels)
    
    print(f"\nAll done! Unique images on disk: {total_unique_files}")

if __name__ == "__main__":
    cache_path = os.path.join(os.path.dirname(OID_BASE_DIR), "fiftyone_cache")
    if os.path.exists(cache_path): shutil.rmtree(cache_path)
    
    prepare_dense_oid_dataset(OID_TRAIN_DIR, samples_per_class=50000)