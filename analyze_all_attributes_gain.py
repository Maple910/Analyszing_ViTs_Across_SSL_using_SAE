# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ==========================================
# ★設定項目
# ==========================================
# 解析対象の属性
TARGET_ATTRIBUTES = [
    "Mobile_phone", "Car", "Guitar", "Sunglasses", "Microphone", 
    "Chair", "Table", "Building", "Person", "Bird", "Tree"
]

# 結果が格納されているルートディレクトリ
RESULTS_ROOT = "./data/final_thesis_results/optimized_classification_separated"

# 保存先
SAVE_DIR = "./data/final_thesis_results/cross_attribute_analysis"

# 有効数字（小数点以下の桁数）
PRECISION = 4
# ==========================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data = []
    
    # テキストレポート用のバッファ
    report_text = ""

    print(">>> Collecting results from all attributes...")
    for attr in TARGET_ATTRIBUTES:
        # CSVファイルを探索 (Maintenance_RateやGainが含まれる最新のファイル)
        csv_path = os.path.join(RESULTS_ROOT, attr, f"opt_accuracy_summary_with_gain_{attr}.csv")
        
        if not os.path.exists(csv_path):
            csv_paths = glob.glob(os.path.join(RESULTS_ROOT, attr, "*.csv"))
            if csv_paths:
                csv_path = max(csv_paths, key=os.path.getmtime)
            else:
                print(f" [Skip] No results found for: {attr}")
                continue

        df = pd.read_csv(csv_path)
        
        for model in df['Model'].unique():
            m_df = df[df['Model'] == model]
            
            try:
                # 数値変換時に精度を確保
                acc_full = float(m_df[m_df['Condition'] == 'Full']['Accuracy'].values[0])
                acc_sae  = float(m_df[m_df['Condition'] == 'SAE-Guided']['Accuracy'].values[0])
                acc_rnd  = float(m_df[m_df['Condition'] == 'Random']['Accuracy'].values[0])
                
                gain = acc_sae - acc_rnd
                m_rate = acc_sae / (acc_full + 1e-8)

                all_data.append({
                    "Attribute": attr,
                    "Model": model,
                    "Full_Acc": round(acc_full, PRECISION),
                    "SAE_Guided_Acc": round(acc_sae, PRECISION),
                    "Random_Acc": round(acc_rnd, PRECISION),
                    "Filtering_Gain": round(gain, PRECISION),
                    "Maintenance_Rate": round(m_rate, PRECISION)
                })
            except Exception as e:
                continue

    master_df = pd.DataFrame(all_data)
    master_df.to_csv(os.path.join(SAVE_DIR, "master_attribute_comparison.csv"), index=False, float_format=f'%.{PRECISION}f')

    # --- 分析1: モデルごとの平均パフォーマンス ---
    summary = master_df.groupby("Model")[["Filtering_Gain", "Random_Acc", "Maintenance_Rate"]].mean()
    # モデルの表示順を固定
    summary = summary.reindex(["MAE", "MoCo v3", "BEiT", "DINO v1"])
    
    header = "\n" + "="*70 + "\n ALL ATTRIBUTES AVERAGE PERFORMANCE (Precision: {})\n".format(PRECISION) + "="*70 + "\n"
    # 小数点以下4桁で揃えて文字列化
    table_str = summary.applymap(lambda x: f"{x:.{PRECISION}f}").to_string() + "\n"
    
    print(header + table_str)
    report_text += header + table_str

    # --- 分析2: 属性別のFiltering Gainヒートマップ ---
    pivot_gain = master_df.pivot(index="Attribute", columns="Model", values="Filtering_Gain")[["MAE", "MoCo v3", "BEiT", "DINO v1"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_gain, annot=True, cmap="YlGnBu", fmt=f".{PRECISION}f")
    plt.title("Filtering Gain (SAE - Random) across Attributes")
    plt.savefig(os.path.join(SAVE_DIR, "heatmap_filtering_gain.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # --- 分析3: 頑健性 (Random Acc) vs 密度 (Filtering Gain) の散布図 ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=master_df, x="Random_Acc", y="Filtering_Gain", hue="Model", style="Model", s=100)
    # 平均Gainを補助線として描画
    plt.axhline(summary["Filtering_Gain"].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.title("Model Characteristics: Robustness vs. Selectivity")
    plt.xlabel("Accuracy with Random Patches (Robustness)")
    plt.ylabel("Improvement by SAE Selection (Filtering Gain)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "robustness_vs_gain_scatter.png"), dpi=200)
    plt.close()

    # --- 結論の自動抽出 ---
    insight_header = "\n>>> Insight Summary (Scientific Profiling):\n"
    print(insight_header)
    report_text += insight_header

    # 全体平均を計算
    mean_gain = summary["Filtering_Gain"].mean()
    mean_rnd = summary["Random_Acc"].mean()

    for model_name, row in summary.iterrows():
        avg_gain = row["Filtering_Gain"]
        avg_rnd = row["Random_Acc"]
        
        char = "Neutral / Balanced"
        if avg_gain > mean_gain and avg_rnd < mean_rnd:
            char = "High Density / Selective (Efficient Focus)"
        elif avg_rnd > mean_rnd:
            char = "High Redundancy / Robust (Distributed Context)"
            
        line = f" - {model_name:<5}: {char:<45} (Avg Gain: {avg_gain:.{PRECISION}f})\n"
        print(line, end="")
        report_text += line

    # テキストファイルへの書き出し
    report_file_path = os.path.join(SAVE_DIR, "attribute_analysis_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n\n [Success] Detailed report saved to: {report_file_path}")
    print(f" [Success] Heatmap and Scatter plots saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()