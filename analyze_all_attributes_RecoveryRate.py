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

# 有効数字（小数点以下の桁数）
PRECISION = 4
# ==========================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data = []
    
    # ターミナル出力を保存するためのテキストバッファ
    report_text = ""

    print(">>> Collecting efficiency metrics from all attributes...")
    for attr in TARGET_ATTRIBUTES:
        # 最新のCSVを探索
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
                
                # Recovery Rate: (SAE - Rnd) / (Full - Rnd)
                recovery = (acc_s - acc_r) / (acc_f - acc_r + 1e-8)
                # Redundancy: Rnd / Full
                redundancy = acc_r / (acc_f + 1e-8)
                # Filtering Gain: SAE - Rnd
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

    # --- 分析1: モデルごとの「平均」解釈効率 ---
    summary = master_df.groupby("Model")[["Recovery_Rate", "Redundancy", "Filtering_Gain"]].mean()
    summary = summary.reindex(["MAE", "MoCo", "BEiT", "DINO"])
    
    header_1 = "\n" + "="*75 + "\n ALL ATTRIBUTES AVERAGE INTERPRETABILITY METRICS\n" + "="*75 + "\n"
    table_str = summary.applymap(lambda x: f"{x:.{PRECISION}f}").to_string() + "\n"
    footer_1 = "-" * 75 + "\n"
    
    # ターミナル表示 & レポート蓄積
    print(header_1 + table_str + footer_1)
    report_text += header_1 + table_str + footer_1

    # --- 結論の自動生成（各モデル個別表示・特性分析版） ---
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

        # 各行の文字列作成
        line1 = f" ■ {model_name:<5}: {profile}\n"
        
        line2_prefix = f"   - Recovery Rate (回復率): {rec:.{PRECISION}f} | "
        if rec > 1.0:
            line2_desc = "驚異的：SAEによる選別がフル画像以上の確信度を生成（ノイズ除去効果）\n"
        else:
            line2_desc = f"SAEによる選別で失われた情報の{rec*100:.2f}%を回復可能\n"
        
        line3_prefix = f"   - Redundancy    (冗長性): {red:.{PRECISION}f} | "
        if red == summary["Redundancy"].min():
            line3_desc = "全モデル中、最も情報の『ごまかし』が効かない純粋な表現\n"
        else:
            line3_desc = f"ランダムな提示でも本来の性能の{red*100:.2f}%を維持可能\n"
            
        line4_prefix = f"   - Filtering Gain (利得) : {gain:.{PRECISION}f} | "
        if gain > avg_gain:
            line4_desc = "SAEの導入による認識精度向上への寄与が平均以上\n"
        else:
            line4_desc = "情報の選別による直接的な精度向上幅\n"
        
        sep = "-" * 70 + "\n"

        # 出力
        model_report = line1 + line2_prefix + line2_desc + line3_prefix + line3_desc + line4_prefix + line4_desc + sep
        print(model_report)
        report_text += model_report
    
    # テキストファイルへの書き出し
    report_file_path = os.path.join(SAVE_DIR, "summary_analysis_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ヒートマップ等の保存
    pivot_recovery = master_df.pivot(index="Attribute", columns="Model", values="Recovery_Rate")[["MAE", "MoCo", "BEiT", "DINO"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_recovery, annot=True, cmap="YlGnBu", fmt=f".{PRECISION}f")
    plt.title("SAE Recovery Rate: Comparison across 11 Attributes")
    plt.savefig(os.path.join(SAVE_DIR, "heatmap_recovery_rate.png"), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n [Success] Text report saved to: {report_file_path}")
    print(f" [Success] Heatmap image saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()