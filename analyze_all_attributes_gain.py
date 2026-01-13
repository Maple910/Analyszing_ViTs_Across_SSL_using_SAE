# 11属性のSAE推定結果を見て分析
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ==========================================
# ★設定項目
# ==========================================
# 解析対象の属性（これまで実行済みのもの）
TARGET_ATTRIBUTES = [
    "Mobile_phone", "Car", "Guitar", "Sunglasses", "Microphone", 
    "Chair", "Table", "Building", "Person", "Bird", "Tree"
]

# 結果が格納されているルートディレクトリ
RESULTS_ROOT = "./data/final_thesis_results/optimized_classification_separated"

# 保存先
SAVE_DIR = "./data/final_thesis_results/cross_attribute_analysis"
# ==========================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data = []

    print(">>> Collecting results from all attributes...")
    for attr in TARGET_ATTRIBUTES:
        # CSVファイルを探索 (Maintenance_RateやGainが含まれる最新のファイル)
        csv_path = os.path.join(RESULTS_ROOT, attr, f"opt_accuracy_summary_with_gain_{attr}.csv")
        
        # もし上記ファイル名でない場合は、別のパターンを探す
        if not os.path.exists(csv_path):
            csv_paths = glob.glob(os.path.join(RESULTS_ROOT, attr, "*.csv"))
            if csv_paths:
                csv_path = max(csv_paths, key=os.path.getmtime)
            else:
                print(f" [Skip] No results found for: {attr}")
                continue

        df = pd.read_csv(csv_path)
        
        # モデルごとの指標を抽出
        for model in df['Model'].unique():
            m_df = df[df['Model'] == model]
            
            # 条件ごとのAccuracy取得
            acc_full = m_df[m_df['Condition'] == 'Full']['Accuracy'].values[0]
            acc_sae  = m_df[m_df['Condition'] == 'SAE-Guided']['Accuracy'].values[0]
            acc_rnd  = m_df[m_df['Condition'] == 'Random']['Accuracy'].values[0]
            
            # Gain（選別利得）を再計算
            gain = acc_sae - acc_rnd
            # 維持率
            m_rate = acc_sae / (acc_full + 1e-8)

            all_data.append({
                "Attribute": attr,
                "Model": model,
                "Full_Acc": acc_full,
                "SAE_Guided_Acc": acc_sae,
                "Random_Acc": acc_rnd,
                "Filtering_Gain": gain,
                "Maintenance_Rate": m_rate
            })

    master_df = pd.DataFrame(all_data)
    master_df.to_csv(os.path.join(SAVE_DIR, "master_attribute_comparison.csv"), index=False)

    # --- 分析1: モデルごとの平均パフォーマンス ---
    summary = master_df.groupby("Model")[["Filtering_Gain", "Random_Acc", "Maintenance_Rate"]].mean()
    print("\n" + "="*60)
    print(" ALL ATTRIBUTES AVERAGE PERFORMANCE")
    print("="*60)
    print(summary.to_string())

    # --- 分析2: 属性別のFiltering Gainヒートマップ ---
    pivot_gain = master_df.pivot(index="Attribute", columns="Model", values="Filtering_Gain")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_gain, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Filtering Gain (SAE - Random) across Attributes")
    plt.savefig(os.path.join(SAVE_DIR, "heatmap_filtering_gain.png"), dpi=200, bbox_inches='tight')

    # --- 分析3: 頑健性 (Random Acc) vs 密度 (Filtering Gain) の散布図 ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=master_df, x="Random_Acc", y="Filtering_Gain", hue="Model", style="Model", s=100)
    plt.axhline(0.15, color='gray', linestyle='--', alpha=0.5)
    plt.title("Model Characteristics: Robustness vs. Selectivity")
    plt.xlabel("Accuracy with Random Patches (Robustness)")
    plt.ylabel("Improvement by SAE Selection (Filtering Gain)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "robustness_vs_gain_scatter.png"), dpi=200)

    # --- 結論の自動抽出 ---
    print("\n>>> Insight Summary:")
    for model in master_df['Model'].unique():
        avg_gain = summary.loc[model, "Filtering_Gain"]
        avg_rnd = summary.loc[model, "Random_Acc"]
        
        char = "Neutral"
        if avg_gain > summary["Filtering_Gain"].mean() and avg_rnd < master_df["Random_Acc"].mean():
            char = "High Density / Selective (Interpretable)"
        elif avg_rnd > master_df["Random_Acc"].mean():
            char = "High Redundancy / Robust"
            
        print(f" - {model:<5}: {char:<40} (Avg Gain: {avg_gain:.3f})")

if __name__ == "__main__":
    main()