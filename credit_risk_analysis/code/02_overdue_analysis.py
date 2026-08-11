# -*- coding: utf-8 -*-
"""
信贷风控数据分析 - 02 逾期行为深度分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')
RESULT_PATH = os.path.join(BASE_DIR, 'results')

df = pd.read_csv(DATA_PATH)
print(f"样本量: {len(df):,}")

# ==========================================
# 1. 逾期程度分层
# ==========================================
print("\n=== 1. 逾期程度分层 ===")

def get_overdue_level(row):
    if row['90天以上逾期次数'] > 0:
        return '严重逾期(90天+)'
    elif row['60-89天逾期次数'] > 0:
        return '中度逾期(60-89天)'
    elif row['30-59天逾期次数'] > 0:
        return '轻度逾期(30-59天)'
    else:
        return '无逾期'

df['逾期程度'] = df.apply(get_overdue_level, axis=1)

overdue_stats = df['逾期程度'].value_counts().reindex(['无逾期', '轻度逾期(30-59天)', '中度逾期(60-89天)', '严重逾期(90天+)'])
overdue_df = pd.DataFrame({
    '人数': overdue_stats.values,
    '占比%': (overdue_stats.values / len(df) * 100).round(2)
}, index=overdue_stats.index)
print(overdue_df)

# 各逾期程度的违约率（90天以上违约标签）
default_by_level = df.groupby('逾期程度', observed=True)['是否违约'].mean() * 100
print(f"\n各逾期程度的90天+违约率:")
print(default_by_level.round(2).sort_values(ascending=False))

# ==========================================
# 2. 不同逾期程度用户特征对比
# ==========================================
print("\n=== 2. 不同逾期程度用户特征对比 ===")

feature_compare = df.groupby('逾期程度', observed=True).agg({
    '年龄': 'mean',
    '月收入': 'mean',
    '负债比率': 'mean',
    '循环贷使用率': 'mean',
    '未结清信贷笔数': 'mean',
    '家属数量': 'mean',
}).round(2)
print(feature_compare)

# ==========================================
# 3. 逾期次数分布
# ==========================================
print("\n=== 3. 30-59天逾期次数分布 ===")
overdue_30 = df['30-59天逾期次数'].value_counts().sort_index().head(10)
overdue_30_df = pd.DataFrame({
    '人数': overdue_30.values,
    '占比%': (overdue_30.values / len(df) * 100).round(2)
}, index=overdue_30.index.astype(int))
print(overdue_30_df)

print(f"\n有过30天以上逾期的用户占比: {(df['30-59天逾期次数'] > 0).mean()*100:.2f}%")
print(f"有过60天以上逾期的用户占比: {(df['60-89天逾期次数'] > 0).mean()*100:.2f}%")
print(f"有过90天以上逾期的用户占比: {(df['90天以上逾期次数'] > 0).mean()*100:.2f}%")

# ==========================================
# 4. 滚动率分析（简化版）
# ==========================================
print("\n=== 4. 滚动率分析（简化版） ===")

# 从轻度逾期滚动到中度的比例
mild_overdue = df[df['30-59天逾期次数'] > 0]
mild_to_moderate = (mild_overdue['60-89天逾期次数'] > 0).mean() * 100
print(f"轻度逾期 → 中度逾期滚动率: {mild_to_moderate:.2f}%")

# 从中度逾期滚动到严重的比例
moderate_overdue = df[df['60-89天逾期次数'] > 0]
moderate_to_severe = (moderate_overdue['90天以上逾期次数'] > 0).mean() * 100
print(f"中度逾期 → 严重逾期滚动率: {moderate_to_severe:.2f}%")

# 从轻度直接滚动到严重的比例
mild_to_severe = (mild_overdue['90天以上逾期次数'] > 0).mean() * 100
print(f"轻度逾期 → 严重逾期滚动率: {mild_to_severe:.2f}%")

# ==========================================
# 可视化
# ==========================================

# 图1：逾期程度分布
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
bars = plt.bar(overdue_stats.index, overdue_stats.values, color=colors, alpha=0.7)
plt.ylabel('人数')
plt.title('用户逾期程度分布')
plt.grid(True, alpha=0.3, axis='y')
for bar, val, pct in zip(bars, overdue_stats.values, overdue_df['占比%']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
             f'{val:,}人\n({pct}%)', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '02_overdue_level.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图2：不同逾期程度特征对比（雷达图简化为柱状图）
features = ['年龄', '月收入(千元)', '负债比率(%)', '循环贷使用率(%)']
normal_vals = [
    feature_compare.loc['无逾期', '年龄'],
    feature_compare.loc['无逾期', '月收入'] / 1000,
    feature_compare.loc['无逾期', '负债比率'] * 100,
    feature_compare.loc['无逾期', '循环贷使用率'] * 100,
]
severe_vals = [
    feature_compare.loc['严重逾期(90天+)', '年龄'],
    feature_compare.loc['严重逾期(90天+)', '月收入'] / 1000,
    feature_compare.loc['严重逾期(90天+)', '负债比率'] * 100,
    feature_compare.loc['严重逾期(90天+)', '循环贷使用率'] * 100,
]

x = np.arange(len(features))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, normal_vals, width, label='无逾期用户', color='#2ecc71', alpha=0.7)
plt.bar(x + width/2, severe_vals, width, label='严重逾期用户', color='#e74c3c', alpha=0.7)
plt.xticks(x, features)
plt.ylabel('数值')
plt.title('正常用户 vs 严重逾期用户 特征对比')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '02_feature_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图3：滚动率漏斗
levels = ['无逾期', '轻度逾期', '中度逾期', '严重逾期']
counts = [
    len(df),
    (df['30-59天逾期次数'] > 0).sum(),
    (df['60-89天逾期次数'] > 0).sum(),
    (df['90天以上逾期次数'] > 0).sum(),
]

plt.figure(figsize=(10, 6))
bars = plt.barh(levels[::-1], counts[::-1], color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71'][::-1], alpha=0.7)
plt.xlabel('人数')
plt.title('逾期滚动漏斗')
plt.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, counts[::-1]):
    plt.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2, 
             f'{val:,}人 ({val/len(df)*100:.2f}%)', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '02_roll_rate_funnel.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n逾期行为分析完成！")
