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
    "MAE":      "#1f77b4",   # 青
    "MoCo v3":  "#ff7f0e",  # オレンジ
    "DINO v1":  "#d62728",  # 赤
    "BEiT":     "#2ca02c"   # 緑
}

MODEL_ORDER = ["MAE", "MoCo v3", "BEiT", "DINO v1"]

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
    audit_header = "="*75 + "\n 1. DATA POINT AUDIT (11 Attributes Check per Model)\n" + "="*75 + "\n"
    report_text += audit_header

    for model in MODEL_ORDER:
        m_df = df[df["Model"] == model]
        found_for_model = m_df["Attribute"].unique()
        missing_for_model = [a for a in EXPECTED_ATTRIBUTES if a not in found_for_model]
        count = len(found_for_model)
        
        status = "SUCCESS" if count == len(EXPECTED_ATTRIBUTES) else "INCOMPLETE"
        line = f" ■ {model:<7}: {count}/{len(EXPECTED_ATTRIBUTES)} points found [{status}]\n"
        if missing_for_model:
            line += f"   - Missing: {', '.join(missing_for_model)}\n"
        report_text += line

    report_text += "\n"

    # --- 2. 相関統計の計算 (Overall & Per Model) ---
    stats_header = "="*75 + "\n 2. CORRELATION STATISTICS (Pearson's r)\n" + "="*75 + "\n"
    report_text += stats_header

    if not df.empty and "Maintenance_Rate" in df.columns:
        # 全体相関
        overall_corr, p_val_all = stats.pearsonr(df["Maintenance_Rate"], df["SAE_Guided_Acc"])
        line_all = f" [OVERALL COMBINED]\n"
        line_all += f"  - Pearson Correlation (r): {overall_corr:.{PRECISION}f}\n"
        line_all += f"  - P-value: {p_val_all:.{PRECISION}e}\n"
        line_all += f"  - Sample size (N): {len(df)}\n\n"
        report_text += line_all

        # モデル別相関
        report_text += " [INDIVIDUAL MODELS]\n"
        for model in MODEL_ORDER:
            m_data = df[df["Model"] == model]
            if len(m_data) > 1:
                r, p = stats.pearsonr(m_data["Maintenance_Rate"], m_data["SAE_Guided_Acc"])
                line_m = f" ■ {model:<7}:\n"
                line_m += f"   - Pearson Correlation (r): {r:.{PRECISION}f}\n"
                line_m += f"   - P-value: {p:.{PRECISION}e}\n"
                line_m += f"   - Sample size (N): {len(m_data)}\n"
            else:
                line_m = f" ■ {model:<7}: Insufficient data to compute correlation (N={len(m_data)})\n"
            report_text += line_m

    # ターミナルにも表示
    print(report_text)

    # --- 3. グラフ描画関数の定義 ---
    def create_scatter(data, title, filename, specific_model=None):
        plt.figure(figsize=(8, 7))
        plt.gca().set_axisbelow(True)
        plt.grid(linestyle='--', alpha=0.4)

        if specific_model:
            # 個別モデルプロット
            plot_data = data[data["Model"] == specific_model]
            sns.scatterplot(data=plot_data, x="Maintenance_Rate", y="SAE_Guided_Acc", 
                            color=MODEL_COLORS[specific_model], marker="o", s=DOT_SIZE, 
                            alpha=1.0, edgecolor='black', linewidth=EDGE_WIDTH,
                            label=f"{specific_model} (N={len(plot_data)})")
        else:
            # 全モデル統合プロット
            sns.scatterplot(data=data, x="Maintenance_Rate", y="SAE_Guided_Acc", 
                            hue="Model", marker="o", s=DOT_SIZE, 
                            hue_order=MODEL_ORDER, palette=MODEL_COLORS, 
                            alpha=1.0, edgecolor='black', linewidth=EDGE_WIDTH)

        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel("Accuracy Maintenance Rate (10% patches / Full)", fontsize=10)
        plt.ylabel("Inference Accuracy (10% patches)", fontsize=10)
        
        # 軸範囲を 0.0 〜 1.0 に固定
        plt.xlim(0.0, 1.1)
        plt.ylim(0.0, 1.0)
        plt.legend(loc='lower right', fontsize=9)
        
        save_path = os.path.join(SAVE_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    # --- 4. 画像生成 ---
    print(">>> Generating correlation plots...")
    create_scatter(df, f"Correlation: Accuracy vs. Maintenance (Combined)\nr = {overall_corr:.4f}", 
                   "correlation_all.png")

    for model in MODEL_ORDER:
        m_data = df[df["Model"] == model]
        if len(m_data) > 1:
            r, _ = stats.pearsonr(m_data["Maintenance_Rate"], m_data["SAE_Guided_Acc"])
            create_scatter(df, f"Correlation: Accuracy vs. Maintenance ({model})\nr = {r:.4f}", 
                          f"correlation_{model.replace(' ', '_')}.png", 
                          specific_model=model)

    # --- 5. レポート保存 ---
    report_file_path = os.path.join(SAVE_DIR, "correlation_analysis_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n [Success] Detailed report and plots saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()