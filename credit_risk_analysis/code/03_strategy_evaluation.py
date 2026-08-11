# -*- coding: utf-8 -*-
"""
信贷风控数据分析 - 03 风控规则策略评估
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
total = len(df)
total_default = df['是否违约'].sum()
base_default_rate = df['是否违约'].mean() * 100

print(f"总样本数: {total:,}")
print(f"整体违约率: {base_default_rate:.2f}%")
print(f"违约样本数: {total_default:,}")

# ==========================================
# 1. 单条规则效果评估
# ==========================================
print("\n=== 1. 单条风控规则效果评估 ===")

rules = [
    ('年龄<22岁直接拒绝', df['年龄'] < 22),
    ('循环贷使用率>90%拒绝', df['循环贷使用率'] > 0.9),
    ('有90天以上逾期记录拒绝', df['90天以上逾期次数'] > 0),
    ('有60天以上逾期记录拒绝', df['60-89天逾期次数'] > 0),
    ('负债比率>100%拒绝', df['负债比率'] > 1),
    ('月收入<2000拒绝', df['月收入'] < 2000),
]

rule_results = []
for rule_name, mask in rules:
    rejected = mask.sum()
    rejected_default = df[mask]['是否违约'].sum()
    pass_rate = (1 - rejected / total) * 100
    # 拒绝的样本中违约的比例
    rejected_default_rate = rejected_default / rejected * 100 if rejected > 0 else 0
    # 通过后的坏账率
    passed_default = total_default - rejected_default
    passed_count = total - rejected
    passed_default_rate = passed_default / passed_count * 100 if passed_count > 0 else 0
    # 坏账率下降幅度
    reduction = (base_default_rate - passed_default_rate) / base_default_rate * 100
    
    rule_results.append({
        '规则名称': rule_name,
        '拒绝人数': rejected,
        '拒绝率%': round(rejected/total*100, 2),
        '通过率%': round(pass_rate, 2),
        '拒绝样本违约率%': round(rejected_default_rate, 2),
        '通过后坏账率%': round(passed_default_rate, 2),
        '坏账率下降%': round(reduction, 2),
    })

rules_df = pd.DataFrame(rule_results)
print(rules_df.to_string(index=False))

# ==========================================
# 2. 组合规则效果（规则集）
# ==========================================
print("\n=== 2. 组合规则策略效果 ===")

strategies = [
    ('宽松策略', 
     (df['年龄'] < 20) | 
     (df['90天以上逾期次数'] > 0)),
    ('中等策略', 
     (df['年龄'] < 22) | 
     (df['循环贷使用率'] > 0.95) |
     (df['90天以上逾期次数'] > 0) |
     (df['负债比率'] > 2)),
    ('严格策略', 
     (df['年龄'] < 25) | 
     (df['循环贷使用率'] > 0.8) |
     (df['60-89天逾期次数'] > 0) |
     (df['90天以上逾期次数'] > 0) |
     (df['负债比率'] > 1) |
     (df['月收入'] < 3000)),
]

strategy_results = []
for name, mask in strategies:
    rejected = mask.sum()
    rejected_default = df[mask]['是否违约'].sum()
    passed_count = total - rejected
    passed_default = total_default - rejected_default
    pass_rate = passed_count / total * 100
    default_rate_after = passed_default / passed_count * 100
    reduction = (base_default_rate - default_rate_after) / base_default_rate * 100
    
    strategy_results.append({
        '策略': name,
        '通过率%': round(pass_rate, 2),
        '拒绝率%': round(rejected/total*100, 2),
        '通过后坏账率%': round(default_rate_after, 2),
        '坏账率下降%': round(reduction, 2),
        '拒绝精准度%': round(rejected_default/rejected*100, 2) if rejected > 0 else 0,
    })

strategies_df = pd.DataFrame(strategy_results)
print(strategies_df.to_string(index=False))

# ==========================================
# 3. 阈值分析（循环贷使用率）
# ==========================================
print("\n=== 3. 循环贷使用率阈值分析 ===")

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
threshold_results = []

for thresh in thresholds:
    mask = df['循环贷使用率'] > thresh
    rejected = mask.sum()
    rejected_default = df[mask]['是否违约'].sum()
    passed_count = total - rejected
    passed_default = total_default - rejected_default
    
    threshold_results.append({
        '阈值': f'>{thresh*100:.0f}%',
        '拒绝率%': round(rejected/total*100, 2),
        '通过率%': round(passed_count/total*100, 2),
        '通过后坏账率%': round(passed_default/passed_count*100, 2),
        '拒绝精准度%': round(rejected_default/rejected*100, 2) if rejected > 0 else 0,
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df.to_string(index=False))

# ==========================================
# 可视化
# ==========================================

# 图1：单条规则效果对比（通过率 vs 坏账率）
plt.figure(figsize=(12, 6))
x = np.arange(len(rules_df))
width = 0.35

plt.bar(x - width/2, rules_df['通过率%'], width, label='通过率', color='#3498db', alpha=0.7)
plt.bar(x + width/2, rules_df['通过后坏账率%'], width, label='通过后坏账率', color='#e74c3c', alpha=0.7)
plt.axhline(y=base_default_rate, color='gray', linestyle='--', label=f'原始坏账率 {base_default_rate:.2f}%')
plt.xticks(x, rules_df['规则名称'], rotation=30, ha='right')
plt.ylabel('百分比 (%)')
plt.title('单条风控规则效果对比')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '03_single_rule_effect.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图2：不同策略的权衡（通过率 vs 坏账率）
plt.figure(figsize=(10, 6))
x = strategies_df['通过率%']
y = strategies_df['通过后坏账率%']
labels = strategies_df['策略']

plt.scatter(x, y, s=200, c=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.7)
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=11, fontweight='bold')

plt.xlabel('通过率 (%)')
plt.ylabel('通过后坏账率 (%)')
plt.title('不同风控策略的通过率-坏账率权衡')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '03_strategy_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图3：阈值分析曲线
plt.figure(figsize=(10, 6))
x = range(len(threshold_df))
plt.plot(x, threshold_df['通过率%'], marker='o', linewidth=2, label='通过率', color='#3498db')
plt.plot(x, threshold_df['通过后坏账率%'], marker='s', linewidth=2, label='通过后坏账率', color='#e74c3c')
plt.plot(x, threshold_df['拒绝精准度%'], marker='^', linewidth=2, label='拒绝精准度', color='#2ecc71')
plt.xticks(x, threshold_df['阈值'])
plt.xlabel('循环贷使用率阈值')
plt.ylabel('百分比 (%)')
plt.title('循环贷使用率阈值效果分析')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '03_threshold_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 保存结果
rules_df.to_csv(os.path.join(RESULT_PATH, 'rule_evaluation.csv'), index=False, encoding='utf-8-sig')
strategies_df.to_csv(os.path.join(RESULT_PATH, 'strategy_comparison.csv'), index=False, encoding='utf-8-sig')

print("\n风控规则策略评估完成！")
