# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
# ==========================================
# 特徴一覧
"""
    "Person"
    "Car"
    "Guitar"
    "Table"
    "Mobile_phone"
    "Bird"
    "Sunglasses"
    "Tree"
    "Building"
    "Chair"
    "Microphone"
"""
# ==========================================
# ==========================================
# ★設定項目
# ==========================================
TARGET_ATTRIBUTE = "Microphone"

# 結果が格納されているディレクトリ
# 以前のスクリプトで生成されたCSVを指定してください
CSV_PATH = f"./data/final_thesis_results/optimized_classification_separated/{TARGET_ATTRIBUTE}/opt_accuracy_summary_with_gain_{TARGET_ATTRIBUTE}.csv"

SAVE_DIR = f"./data/final_thesis_results/interpretability_analysis/{TARGET_ATTRIBUTE}"
# ==========================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        print(f" [Error] CSVが見つかりません: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # モデルごとに集計
    models = df['Model'].unique()
    analysis_data = []

    for model in models:
        m_df = df[df['Model'] == model]
        
        acc_full = m_df[m_df['Condition'] == 'Full']['Accuracy'].values[0]
        acc_sae  = m_df[m_df['Condition'] == 'SAE-Guided']['Accuracy'].values[0]
        acc_rnd  = m_df[m_df['Condition'] == 'Random']['Accuracy'].values[0]
        
        # 1. Filtering Gain (純粋な選別利得)
        gain = acc_sae - acc_rnd
        
        # 2. Recovery Rate (解釈効率)
        # 「知らない状態」から「全部知っている状態」の差を、SAEがどれだけ埋めたか
        recovery = (acc_sae - acc_rnd) / (acc_full - acc_rnd + 1e-8)
        
        # 3. Redundancy (冗長性)
        # ランダムでも当たってしまう割合
        redundancy = acc_rnd / (acc_full + 1e-8)

        analysis_data.append({
            "Model": model,
            "Full_Acc": acc_full,
            "SAE_Guided_Acc": acc_sae,
            "Random_Acc": acc_rnd,
            "Filtering_Gain": round(gain, 3),
            "Recovery_Rate": round(recovery, 3),
            "Redundancy": round(redundancy, 3)
        })

    res_df = pd.DataFrame(analysis_data)
    res_df.to_csv(os.path.join(SAVE_DIR, "interpretable_efficiency_report.csv"), index=False)

    # --- 可視化1: Recovery Rate の比較 ---
    plt.figure(figsize=(10, 6))
    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c']
    bars = plt.bar(res_df['Model'], res_df['Recovery_Rate'], color=colors, alpha=0.8, edgecolor='black')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.1%}", ha='center', va='bottom', fontweight='bold')
    
    plt.title(f"SAE Recovery Rate: {TARGET_ATTRIBUTE}\n(How much gap was filled by SAE-identified patches?)", fontsize=12)
    plt.ylabel("Recovery Rate (0.0 - 1.0)"); plt.ylim(0, 1.2); plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "recovery_rate_comparison.png"), dpi=200)

    # --- 可視化2: 冗長性 vs 選別利得 ---
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=res_df, x="Redundancy", y="Filtering_Gain", hue="Model", s=200, style="Model")
    plt.axhline(0.10, color='gray', linestyle='--', alpha=0.5)
    plt.title("Model Profile: Redundancy vs. Selectivity")
    plt.xlabel("Redundancy (Random Accuracy / Full Accuracy)"); plt.ylabel("Filtering Gain (SAE - Random)")
    
    # 領域の説明を追加
    plt.text(res_df['Redundancy'].max()*0.9, res_df['Filtering_Gain'].max()*1.1, "Interpretable High-Density", color='blue', fontweight='bold')
    plt.text(res_df['Redundancy'].max()*0.9, 0.05, "Robust / Redundant", color='red', fontweight='bold')
    
    plt.savefig(os.path.join(SAVE_DIR, "model_profile_scatter.png"), dpi=200)

    print("\n" + "="*80)
    print(f" INTERPRETABILITY ANALYSIS REPORT: {TARGET_ATTRIBUTE}")
    print("="*80)
    print(res_df.to_string(index=False))
    print("-" * 80)
    print(" [Insight]")
    
    # 自動分析コメント
    mae_rec = res_df[res_df['Model'] == 'MAE']['Recovery_Rate'].values[0]
    beit_red = res_df[res_df['Model'] == 'BEiT']['Redundancy'].values[0]
    
    print(f" * MAE's Recovery Rate is {mae_rec:.1%}. This shows high selectivity.")
    print(f" * BEiT's Redundancy is {beit_red:.1%}. This indicates high information spread.")
    print("="*80)

if __name__ == "__main__":
    main()