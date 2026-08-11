# -*- coding: utf-8 -*-
"""
信贷风控数据分析 - 04 策略优化建议与总结
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
base_default_rate = df['是否违约'].mean() * 100

print("=" * 60)
print("信贷风控数据分析 - 核心发现与策略建议")
print("=" * 60)

# ==========================================
# 核心发现总结
# ==========================================
print("\n【一、核心发现】")

print("\n1. 用户画像方面：")
print("   - 年龄是最强的风险分层变量：18-25岁违约率11.17%，65岁以上仅2.41%")
print("   - 收入与违约率负相关：3k以下违约率9.43%，12k以上约4.5%")
print("   - 信贷笔数呈U型关系：0-2笔新人违约率最高（12.92%），6-8笔最低（5.35%）")

print("\n2. 逾期行为方面：")
print("   - 20%的用户有过不同程度的逾期，其中5.56%达到严重逾期（90天+）")
print("   - 滚动率：轻度→中度18.3%，中度→严重36.7%（中度逾期是催收关键节点）")
print("   - 严重逾期用户特征：更年轻、收入更低、循环贷使用率更高")

print("\n3. 规则策略方面：")
print("   - 最强单规则：循环贷使用率>90%（拒绝13%用户，坏账率下降36%）")
print("   - 次强规则：有90天以上逾期记录（拒绝5.6%用户，坏账率下降31%）")
print("   - 负债比率单独使用效果差，需要和其他变量组合")

# ==========================================
# 策略建议
# ==========================================
print("\n【二、策略优化建议】")

print("\n建议1：审批端 - 分层准入策略")
print("   - 高风险人群（年龄<25岁 + 循环贷使用率>80%）：人工审核或降额")
print("   - 中风险人群：正常审批，额度适中")
print("   - 低风险人群（年龄>50岁 + 无逾期记录）：自动审批，可提额")
print("   - 预期效果：整体坏账率降低20-30%，通过率保持在80%以上")

print("\n建议2：贷中管理 - 重点关注中度逾期客户")
print("   - 中度逾期（60-89天）客户滚入严重逾期的比例高达36.7%")
print("   - 建议在60天节点加大催收力度，提前介入")
print("   - 对有还款意愿的客户提供分期或展期方案")
print("   - 预期效果：严重逾期率降低15-20%")

print("\n建议3：新用户风险管理")
print("   - 0-2笔信贷记录的新人违约率高达12.92%（信用白户风险高）")
print("   - 建议新用户首笔额度降低，逐步提额")
print("   - 加强首贷前的信息验证和反欺诈核查")
print("   - 预期效果：新用户坏账率降低25%")

print("\n建议4：规则优化 - 用组合规则替代单规则")
print("   - 单条负债比率规则效果差（拒绝精准度仅6.5%）")
print("   - 建议：负债比率>100% 且 循环贷使用率>70% 才拒绝")
print("   - 可大幅降低误杀率，提升拒绝精准度")

# ==========================================
# 风险矩阵
# ==========================================
print("\n【三、用户风险分层矩阵】")

# 简单的风险分层
df['风险等级'] = '低风险'
df.loc[(df['循环贷使用率'] > 0.7) | (df['30-59天逾期次数'] > 0), '风险等级'] = '中风险'
df.loc[(df['循环贷使用率'] > 0.9) | (df['90天以上逾期次数'] > 0), '风险等级'] = '高风险'

risk_stats = df.groupby('风险等级', observed=True).agg(
    人数=('是否违约', 'count'),
    违约数=('是否违约', 'sum'),
).reset_index()
risk_stats['占比%'] = (risk_stats['人数'] / total * 100).round(2)
risk_stats['违约率%'] = (risk_stats['违约数'] / risk_stats['人数'] * 100).round(2)
risk_stats = risk_stats.set_index('风险等级').reindex(['低风险', '中风险', '高风险'])
print(risk_stats)

# ==========================================
# 可视化
# ==========================================

# 图1：核心发现总结图（4个关键指标）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 年龄vs违约率
ax = axes[0, 0]
age_bins = [18, 25, 35, 45, 55, 65, 100]
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
df['年龄段'] = pd.cut(df['年龄'], bins=age_bins, labels=age_labels)
age_default = df.groupby('年龄段', observed=True)['是否违约'].mean() * 100
ax.bar(age_labels, age_default.values, color='#e74c3c', alpha=0.7)
ax.set_title('不同年龄段违约率')
ax.set_ylabel('违约率 (%)')
ax.grid(True, alpha=0.3, axis='y')

# 循环贷使用率vs违约率
ax = axes[0, 1]
util_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0, 10]
util_labels = ['<30%', '30-50%', '50-70%', '70-90%', '90-100%', '>100%']
df['使用率段'] = pd.cut(df['循环贷使用率'], bins=util_bins, labels=util_labels)
util_default = df.groupby('使用率段', observed=True)['是否违约'].mean() * 100
ax.bar(util_labels, util_default.values, color='#3498db', alpha=0.7)
ax.set_title('循环贷使用率与违约率')
ax.set_ylabel('违约率 (%)')
ax.tick_params(axis='x', rotation=30)
ax.grid(True, alpha=0.3, axis='y')

# 风险分层
ax = axes[1, 0]
risk_order = ['低风险', '中风险', '高风险']
risk_counts = [risk_stats.loc[r, '人数'] for r in risk_order]
risk_defaults = [risk_stats.loc[r, '违约率%'] for r in risk_order]
colors = ['#2ecc71', '#f1c40f', '#e74c3c']
bars = ax.bar(risk_order, risk_counts, color=colors, alpha=0.7)
ax.set_title('用户风险分层分布')
ax.set_ylabel('人数')
ax2 = ax.twinx()
ax2.plot(risk_order, risk_defaults, 'o-', color='black', linewidth=2, markersize=8)
ax2.set_ylabel('违约率 (%)')
ax.grid(True, alpha=0.3, axis='y')

# 策略权衡曲线
ax = axes[1, 1]
pass_rates = [94.4, 86.6, 67.8, 51.8]
default_rates = [4.63, 4.26, 3.86, 2.66]
labels = ['宽松策略', '循环贷>90%', '中等策略', '严格策略']
ax.plot(pass_rates, default_rates, 'o-', linewidth=2, markersize=8, color='#9b59b6')
for i, label in enumerate(labels):
    ax.annotate(label, (pass_rates[i], default_rates[i]), 
                textcoords="offset points", xytext=(5,5), fontsize=9)
ax.axhline(y=base_default_rate, color='gray', linestyle='--', alpha=0.5, label=f'原始坏账率 {base_default_rate:.2f}%')
ax.set_xlabel('通过率 (%)')
ax.set_ylabel('通过后坏账率 (%)')
ax.set_title('通过率-坏账率权衡曲线')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '04_summary_dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()

# 保存风险分层结果
df.to_csv(os.path.join(BASE_DIR, 'data', 'risk_scored_data.csv'), index=False, encoding='utf-8-sig')

print("\n策略优化分析完成！")
print(f"\n所有结果图表已保存到 results/ 目录")
