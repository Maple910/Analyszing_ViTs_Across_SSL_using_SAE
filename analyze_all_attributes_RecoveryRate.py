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
# 解析対象の属性（11属性）
TARGET_ATTRIBUTES = [
    "Mobile_phone", "Car", "Guitar", "Sunglasses", "Microphone", 
    "Chair", "Table", "Building", "Person", "Bird", "Tree"
]

# 精度結果が格納されているルートディレクトリ
RESULTS_ROOT = "./data/final_thesis_results/optimized_classification_separated"

# 保存先
SAVE_DIR = "./data/final_thesis_results/cross_attribute_efficiency"

# 有効数字
PRECISION = 4

# 表示順序の定義
MODEL_ORDER = ["MAE", "MoCo v3", "BEiT", "DINO v1"]
# ==========================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data = []
    
    report_text = ""

    print(">>> Collecting efficiency metrics from all attributes...")
    for attr in TARGET_ATTRIBUTES:
        pattern = os.path.join(RESULTS_ROOT, attr, f"opt_accuracy_summary_with_gain_{attr}.csv")
        csv_paths = glob.glob(pattern)
        if not csv_paths:
            csv_paths = glob.glob(os.path.join(RESULTS_ROOT, attr, "*.csv"))
        
        if not csv_paths:
            continue

        csv_path = max(csv_paths, key=os.path.getmtime)
        df = pd.read_csv(csv_path)
        
        for model in df['Model'].unique():
            m_df = df[df['Model'] == model]
            try:
                acc_f = float(m_df[m_df['Condition'] == 'Full']['Accuracy'].values[0])
                acc_s = float(m_df[m_df['Condition'] == 'SAE-Guided']['Accuracy'].values[0])
                acc_r = float(m_df[m_df['Condition'] == 'Random']['Accuracy'].values[0])
                
                recovery = (acc_s - acc_r) / (acc_f - acc_r + 1e-8)
                redundancy = acc_r / (acc_f + 1e-8)
                gain = acc_s - acc_r

                all_data.append({
                    "Attribute": attr, "Model": model,
                    "Recovery_Rate": round(recovery, PRECISION), 
                    "Redundancy": round(redundancy, PRECISION),
                    "Filtering_Gain": round(gain, PRECISION), 
                    "Full_Acc": round(acc_f, PRECISION)
                })
            except: continue

    master_df = pd.DataFrame(all_data)
    master_df.to_csv(os.path.join(SAVE_DIR, "master_efficiency_comparison.csv"), index=False, float_format=f'%.{PRECISION}f')

    # --- 分析1: モデルごとの「平均」解釈効率 (再ソート適用) ---
    summary = master_df.groupby("Model")[["Recovery_Rate", "Redundancy", "Filtering_Gain"]].mean()
    summary = summary.reindex(MODEL_ORDER)
    
    header_1 = "\n" + "="*75 + "\n ALL ATTRIBUTES AVERAGE INTERPRETABILITY METRICS\n" + "="*75 + "\n"
    table_str = summary.applymap(lambda x: f"{x:.{PRECISION}f}").to_string() + "\n"
    footer_1 = "-" * 75 + "\n"
    
    print(header_1 + table_str + footer_1)
    report_text += header_1 + table_str + footer_1

    # --- 結論の自動生成 ---
    avg_rec = summary["Recovery_Rate"].mean()
    avg_red = summary["Redundancy"].mean()
    avg_gain = summary["Filtering_Gain"].mean()

    insight_header = "\n>>> Model Specific Characteristics Analysis (Scientific Insight):\n"
    print(insight_header)
    report_text += insight_header

    for model_name, row in summary.iterrows():
        rec = row['Recovery_Rate']
        red = row['Redundancy']
        gain = row['Filtering_Gain']
        
        traits = []
        if red < avg_red: traits.append("情報の高密度性（Pure）")
        if rec > 1.0: traits.append("情報の純化・超回復（Clarified）")
        elif rec > avg_rec: traits.append("高い解釈効率（Efficient）")
        if gain > avg_gain: traits.append("選別による高利得（High-Gain）")
        if red > avg_red: traits.append("高い頑健性・冗長性（Robust）")

        profile = " / ".join(traits) if traits else "標準的表現（Balanced）"

        line1 = f" ■ {model_name:<7}: {profile}\n"
        line2 = f"   - Recovery Rate (回復率): {rec:.{PRECISION}f}\n"
        line3 = f"   - Redundancy    (冗長性): {red:.{PRECISION}f}\n"
        line4 = f"   - Filtering Gain (利得) : {gain:.{PRECISION}f}\n"
        sep = "-" * 70 + "\n"

        model_report = line1 + line2 + line3 + line4 + sep
        print(model_report)
        report_text += model_report
    
    report_file_path = os.path.join(SAVE_DIR, "summary_analysis_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ヒートマップ等の保存 (列順序を指定)
    pivot_recovery = master_df.pivot(index="Attribute", columns="Model", values="Recovery_Rate")[MODEL_ORDER]
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_recovery, annot=True, cmap="YlGnBu", fmt=f".{PRECISION}f")
    plt.title("SAE Recovery Rate: Comparison across 11 Attributes")
    plt.savefig(os.path.join(SAVE_DIR, "heatmap_recovery_rate.png"), dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()