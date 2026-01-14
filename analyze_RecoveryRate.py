# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# ==========================================
# 特徴一覧
"""
    "Person", "Car", "Guitar", "Table", "Mobile_phone",
    "Bird", "Sunglasses", "Tree", "Building", "Chair", "Microphone"
"""
# ==========================================

# ==========================================
# ★設定項目
# ==========================================
TARGET_ATTRIBUTE = "Microphone"

# 結果が格納されているディレクトリ
CSV_PATH = f"./data/final_thesis_results/optimized_classification_separated/{TARGET_ATTRIBUTE}/opt_accuracy_summary_with_gain_{TARGET_ATTRIBUTE}.csv"

SAVE_DIR = f"./data/final_thesis_results/interpretability_analysis/{TARGET_ATTRIBUTE}"

# 有効数字（小数点以下の桁数）
PRECISION = 4
# ==========================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    report_text = ""  # テキスト保存用のバッファ
    
    if not os.path.exists(CSV_PATH):
        print(f" [Error] CSVが見つかりません: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # モデルごとに集計
    models = df['Model'].unique()
    analysis_data = []

    for model in models:
        m_df = df[df['Model'] == model]
        
        try:
            acc_full = float(m_df[m_df['Condition'] == 'Full']['Accuracy'].values[0])
            acc_sae  = float(m_df[m_df['Condition'] == 'SAE-Guided']['Accuracy'].values[0])
            acc_rnd  = float(m_df[m_df['Condition'] == 'Random']['Accuracy'].values[0])
            
            # 1. Filtering Gain (純粋な選別利得)
            gain = acc_sae - acc_rnd
            
            # 2. Recovery Rate (解釈効率)
            recovery = (acc_sae - acc_rnd) / (acc_full - acc_rnd + 1e-8)
            
            # 3. Redundancy (冗長性)
            redundancy = acc_rnd / (acc_full + 1e-8)

            analysis_data.append({
                "Model": model,
                "Full_Acc": round(acc_full, PRECISION),
                "SAE_Guided_Acc": round(acc_sae, PRECISION),
                "Random_Acc": round(acc_rnd, PRECISION),
                "Filtering_Gain": round(gain, PRECISION),
                "Recovery_Rate": round(recovery, PRECISION),
                "Redundancy": round(redundancy, PRECISION)
            })
        except: continue

    res_df = pd.DataFrame(analysis_data)
    # 順序を揃える
    res_df = res_df.set_index("Model").reindex(["MAE", "MoCo v3", "BEiT", "DINO v1"]).reset_index()
    res_df.to_csv(os.path.join(SAVE_DIR, "interpretable_efficiency_report.csv"), index=False, float_format=f'%.{PRECISION}f')

    # --- 可視化1: Recovery Rate の比較 (不透明・描画順修正) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True) # グリッドを棒の後ろに配置
    
    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c']
    # alpha=1.0 (不透明) に設定
    bars = ax.bar(res_df['Model'], res_df['Recovery_Rate'], color=colors, alpha=1.0, edgecolor='black', width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                 f"{height:.{PRECISION}f}", ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(f"SAE Recovery Rate: {TARGET_ATTRIBUTE}\n(Performance Gap Filled by SAE Selection)", fontsize=12)
    ax.set_ylabel(f"Recovery Rate (Precision: {PRECISION})")
    ax.set_ylim(0, max(res_df['Recovery_Rate'].max() * 1.2, 1.2))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.savefig(os.path.join(SAVE_DIR, "recovery_rate_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # --- 可視化2: 冗長性 vs 選別利得 (点にかぶるテキストラベルを削除) ---
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=res_df, x="Redundancy", y="Filtering_Gain", hue="Model", s=200, style="Model")
    plt.axhline(res_df['Filtering_Gain'].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Model Profile: Redundancy vs. Selectivity ({TARGET_ATTRIBUTE})")
    plt.xlabel("Redundancy (Random / Full)"); plt.ylabel("Filtering Gain (SAE - Random)")
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "model_profile_scatter.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # --- レポート作成 ---
    header = "\n" + "="*85 + "\n"
    header += f" INTERPRETABILITY ANALYSIS REPORT: {TARGET_ATTRIBUTE}\n"
    header += "="*85 + "\n"
    
    # テーブルの整形
    table_str = res_df.applymap(lambda x: f"{x:.{PRECISION}f}" if isinstance(x, (int, float)) else x).to_string(index=False) + "\n"
    
    insight_header = "-" * 85 + "\n [Insight Analysis]\n"
    
    # 自動分析コメント
    mae_row = res_df[res_df['Model'] == 'MAE']
    beit_row = res_df[res_df['Model'] == 'BEiT']
    
    insights = ""
    if not mae_row.empty:
        mae_rec = mae_row['Recovery_Rate'].values[0]
        insights += f" * MAE's Recovery Rate is {mae_rec:.{PRECISION}f}. This indicates how effectively the SAE isolates critical patches.\n"
    if not beit_row.empty:
        beit_red = beit_row['Redundancy'].values[0]
        insights += f" * BEiT's Redundancy is {beit_red:.{PRECISION}f}. This reflects the model's ability to infer from context.\n"
    
    footer = "="*85 + "\n"

    report_text = header + table_str + insight_header + insights + footer
    print(report_text)
    
    report_file_path = os.path.join(SAVE_DIR, f"efficiency_detailed_report_{TARGET_ATTRIBUTE}.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f" [Success] Bar transparency issues resolved (alpha=1.0, set_axisbelow=True).")
    print(f" [Success] Detailed analysis report saved to: {report_file_path}")

if __name__ == "__main__":
    main()