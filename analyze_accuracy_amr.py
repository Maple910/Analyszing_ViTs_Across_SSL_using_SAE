# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# ==========================================
# ★設定項目
# ==========================================
# analyze_attribute_performance.py で生成された全属性統合CSVを使用
INPUT_CSV = "./data/final_thesis_results/cross_attribute_analysis/master_attribute_comparison.csv"

# 保存先
SAVE_DIR = "./data/final_thesis_results/correlation_analysis"

# 有効数字
PRECISION = 4

# チェック対象の全11属性リスト
EXPECTED_ATTRIBUTES = [
    "Mobile_phone", "Car", "Guitar", "Sunglasses", "Microphone", 
    "Chair", "Table", "Building", "Person", "Bird", "Tree"
]

# 論文用：統一カラー設定
MODEL_COLORS = {
    "MAE":  "#1f77b4",   # 青
    "MoCo": "#ff7f0e",  # オレンジ
    "DINO": "#d62728",  # 赤
    "BEiT": "#2ca02c"   # 緑
}

MODEL_ORDER = ["MAE", "MoCo", "DINO", "BEiT"]

# ドットのサイズ
DOT_SIZE = 60
# 縁取りの線幅
EDGE_WIDTH = 0.6
# ==========================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    report_text = ""

    if not os.path.exists(INPUT_CSV):
        print(f" [Error] 基となるCSVが見つかりません: {INPUT_CSV}")
        return

    # データの読み込み
    df = pd.read_csv(INPUT_CSV)
    
    # --- 1. モデル別属性網羅性チェック (Audit) ---
    audit_results = []
    audit_header = "="*75 + "\n DATA POINT AUDIT (11 Attributes Check per Model)\n" + "="*75 + "\n"
    print(audit_header)
    report_text += audit_header

    for model in MODEL_ORDER:
        m_df = df[df["Model"] == model]
        found_for_model = m_df["Attribute"].unique()
        missing_for_model = [a for a in EXPECTED_ATTRIBUTES if a not in found_for_model]
        count = len(found_for_model)
        
        status = "SUCCESS" if count == len(EXPECTED_ATTRIBUTES) else "INCOMPLETE"
        line = f" ■ {model:<5}: {count}/{len(EXPECTED_ATTRIBUTES)} points found [{status}]\n"
        if missing_for_model:
            line += f"   - Missing: {', '.join(missing_for_model)}\n"
        
        print(line, end="")
        report_text += line
        audit_results.append({"Model": model, "Found": count, "Status": status})

    sep = "-" * 75 + "\n"
    print(sep)
    report_text += sep

    # --- 2. 相関統計の計算 ---
    overall_corr, p_value = stats.pearsonr(df["Maintenance_Rate"], df["SAE_Guided_Acc"])
    
    stats_header = " CORRELATION STATISTICS (Across all valid data points)\n" + "-"*75 + "\n"
    overall_stats = f" Overall Pearson Correlation (r): {overall_corr:.{PRECISION}f}\n"
    overall_stats += f" P-value: {p_value:.{PRECISION}e}\n\n"
    report_text += stats_header + overall_stats
    print(stats_header + overall_stats)

    # --- 3. グラフ描画関数の定義 ---
    def create_scatter(data, title, filename, specific_model=None):
        plt.figure(figsize=(8, 7))
        plt.gca().set_axisbelow(True)
        plt.grid(linestyle='--', alpha=0.4)

        # 背景に全体傾向の回帰直線
        sns.regplot(data=data, x="Maintenance_Rate", y="SAE_Guided_Acc", 
                    scatter=False, color="black", 
                    line_kws={"linestyle": "--", "alpha": 0.2, "linewidth": 1.2})

        if specific_model:
            # 個別モデルプロット (縁取り追加)
            plot_data = data[data["Model"] == specific_model]
            sns.scatterplot(data=plot_data, x="Maintenance_Rate", y="SAE_Guided_Acc", 
                            color=MODEL_COLORS[specific_model], marker="o", s=DOT_SIZE, 
                            alpha=1.0, edgecolor='black', linewidth=EDGE_WIDTH,
                            label=f"{specific_model} (N={len(plot_data)})")
        else:
            # 全モデル統合プロット (縁取り追加)
            sns.scatterplot(data=data, x="Maintenance_Rate", y="SAE_Guided_Acc", 
                            hue="Model", marker="o", s=DOT_SIZE, 
                            hue_order=MODEL_ORDER, palette=MODEL_COLORS, 
                            alpha=1.0, edgecolor='black', linewidth=EDGE_WIDTH)

        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel("Accuracy Maintenance Rate (10% patches / Full)", fontsize=10)
        plt.ylabel("Inference Accuracy (10% patches)", fontsize=10)
        plt.xlim(-0.02, 1.05)
        plt.ylim(-0.02, 1.05)
        plt.legend(loc='lower right', fontsize=9)
        
        save_path = os.path.join(SAVE_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    # --- 4. 計5枚の画像生成 ---
    print(">>> Generating correlation plots...")

    # ① 全モデル統合版
    path_all = create_scatter(df, f"Correlation: Accuracy vs. Maintenance (Combined)\nOverall r = {overall_corr:.4f}", 
                              "correlation_acc_vs_maintenance_all.png")
    print(f"  [Saved] {os.path.basename(path_all)}")

    # ②-⑤ 個別モデル版
    for model in MODEL_ORDER:
        m_data = df[df["Model"] == model]
        if m_data.empty: continue
        
        m_corr, _ = stats.pearsonr(m_data["Maintenance_Rate"], m_data["SAE_Guided_Acc"])
        m_info = f" - {model:<5}: Pearson r = {m_corr:.{PRECISION}f}\n"
        report_text += m_info
        
        path_indiv = create_scatter(df, f"Correlation: Accuracy vs. Maintenance ({model})\nModel r = {m_corr:.4f}", 
                                    f"correlation_acc_vs_maintenance_{model}.png", 
                                    specific_model=model)
        print(f"  [Saved] {os.path.basename(path_indiv)}")

    # --- 5. レポート保存 ---
    report_file = os.path.join(SAVE_DIR, "correlation_analysis_audit_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n [Success] Audit complete and 5 plots (with black edges) saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()