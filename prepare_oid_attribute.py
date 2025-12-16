# prepare_oid_attribute.py
# 可視化の際に平均スコアを取るためのpos/negデータセット作成
import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil
from PIL import Image
from config_oid import OID_BASE_DIR, TARGET_ATTRIBUTE

def resize_and_save(src_path, dest_path, size=256):
    """画像をリサイズして保存するヘルパー関数"""
    try:
        with Image.open(src_path) as img:
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
    except Exception:
        pass

def process_and_cleanup(dataset_or_view, output_dir, cache_dir):
    """データセット(またはView)をエクスポートし、リサイズ保存してから元データを消す"""
    temp_dir = os.path.join(cache_dir, "temp_attr_export")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    
    # 画像のエクスポート
    dataset_or_view.export(export_dir=temp_dir, dataset_type=fo.types.ImageDirectory)
    
    # 画像ファイルを取得してリサイズ保存
    saved_count = 0
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(root, file)
                dest = os.path.join(output_dir, file)
                resize_and_save(src, dest)
                saved_count += 1
    
    # 一時フォルダ削除
    shutil.rmtree(temp_dir)
    
    # Validationデータのキャッシュ削除
    val_cache = os.path.join(cache_dir, "open-images-v7", "validation", "data")
    if os.path.exists(val_cache):
        shutil.rmtree(val_cache)
        os.makedirs(val_cache, exist_ok=True)
        
    return saved_count

def save_attribute_stats(target_class, base_dir, attr_dir, non_attr_dir):
    """最終的なデータセットの統計情報をテキストファイルに保存する"""
    # ディスク上の実ファイル数をカウント
    pos_count = len([f for f in os.listdir(attr_dir) if f.lower().endswith('.jpg')])
    neg_count = len([f for f in os.listdir(non_attr_dir) if f.lower().endswith('.jpg')])
    total = pos_count + neg_count
    
    stats_path = os.path.join(base_dir, target_class, "dataset_stats.txt")
    
    with open(stats_path, "w") as f:
        f.write(f"=== Analysis Dataset Statistics: {target_class} ===\n")
        f.write(f"Target Attribute: {target_class}\n\n")
        f.write(f"Positive (Attribute Present): {pos_count} images\n")
        f.write(f"Negative (Attribute Absent) : {neg_count} images\n")
        f.write(f"  * Note: Negative set explicitly excludes images labeled as '{target_class}'.\n")
        f.write(f"------------------------------------------\n")
        f.write(f"Total Images                : {total} images\n")
        
    print(f"\n[Stats] Statistics saved to: {stats_path}")
    print(f"        Positive: {pos_count}, Negative: {neg_count}, Total: {total}")

def prepare_attribute_dataset(target_class, base_dir, max_samples=2000):
    cache_dir = os.path.join(os.path.dirname(OID_BASE_DIR), "fiftyone_cache")
    fo.config.dataset_zoo_dir = cache_dir
    
    print(f"--- Preparing Analysis Dataset for attribute: {target_class} ---")
    print(f"Target count: {max_samples} images for each (Positive/Negative)")
    
    attr_dir = os.path.join(base_dir, target_class, "positive")
    non_attr_dir = os.path.join(base_dir, target_class, "negative")
    os.makedirs(attr_dir, exist_ok=True)
    os.makedirs(non_attr_dir, exist_ok=True)

    # 1. Positive (変更なし)
    current_pos = len(os.listdir(attr_dir))
    if current_pos < max_samples:
        print(f"Downloading Positive images...")
        dataset_pos = foz.load_zoo_dataset(
            "open-images-v7", split="train", label_types=["detections"],
            classes=[target_class], max_samples=max_samples, shuffle=True, seed=42,
            dataset_name="pos_loader", drop_existing_dataset=True
        )
        process_and_cleanup(dataset_pos, attr_dir, cache_dir)
        dataset_pos.delete()
    else:
        print(f"Positive images already exist ({current_pos} images).")

    # 2. Negative (try-exceptによる安全なフィルタリング)
    current_neg = len(os.listdir(non_attr_dir))
    if current_neg < max_samples:
        print(f"Downloading Negative images (Excluding '{target_class}')...")
        
        # 除外で減る分を見越して 2倍 のバッファを取る
        buffer_size = int(max_samples * 2.0)
        
        dataset_neg = foz.load_zoo_dataset(
            "open-images-v7", split="train", label_types=["detections"],
            max_samples=buffer_size, shuffle=True, seed=99,
            dataset_name="neg_loader", drop_existing_dataset=True
        )
        
        # --- 修正: try-except で安全にアクセス ---
        valid_ids = []
        excluded_count = 0
        
        print("  -> Filtering images in Python loop...")
        for sample in dataset_neg:
            try:
                # フィールドへのアクセスを試みる
                dets = sample.detections
            except AttributeError:
                # フィールドが存在しない場合はNone扱い
                dets = None
            
            # パターンA: detection自体がない (None) -> 安全 (ギターではない)
            if dets is None:
                valid_ids.append(sample.id)
                continue
            
            # パターンB: detectionはある -> ラベルを確認
            labels = [d.label for d in dets.detections]
            
            if target_class in labels:
                # ターゲットが含まれているので除外
                excluded_count += 1
            else:
                # ターゲットが含まれていないので採用
                valid_ids.append(sample.id)
        
        # IDリストを使ってViewを作成
        final_view = dataset_neg.select(valid_ids).limit(max_samples)
        
        print(f"  -> Downloaded buffer: {len(dataset_neg)}")
        print(f"  -> Excluded (contains '{target_class}'): {excluded_count}")
        print(f"  -> Remaining candidates: {len(valid_ids)}")
        print(f"  -> Trimming to target: {max_samples}")
        
        process_and_cleanup(final_view, non_attr_dir, cache_dir)
        dataset_neg.delete()
    else:
        print(f"Negative images already exist ({current_neg} images).")
    
    # 3. 統計情報の保存
    save_attribute_stats(target_class, base_dir, attr_dir, non_attr_dir)
    
    print(f"\nAnalysis data preparation complete for {target_class}!")

if __name__ == "__main__":
    os.makedirs(OID_BASE_DIR, exist_ok=True)
    prepare_attribute_dataset(TARGET_ATTRIBUTE, OID_BASE_DIR, max_samples=2000)