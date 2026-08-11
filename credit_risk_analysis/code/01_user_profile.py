# -*- coding: utf-8 -*-
"""
信贷风控数据分析 - 01 数据概览与用户画像
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw_data.csv')
RESULT_PATH = os.path.join(BASE_DIR, 'results')

# 读取数据
df = pd.read_csv(DATA_PATH, index_col=0)
df.columns = ['是否违约', '循环贷使用率', '年龄', '30-59天逾期次数', '负债比率', 
              '月收入', '未结清信贷笔数', '90天以上逾期次数', '不动产贷款笔数', 
              '60-89天逾期次数', '家属数量']

print(f"样本量: {len(df):,}")
print(f"违约率: {df['是否违约'].mean()*100:.2f}%")
print(f"违约样本: {df['是否违约'].sum():,}, 正常样本: {(df['是否违约']==0).sum():,}")

# 数据清洗（简单处理）
df = df[df['年龄'] >= 18].copy()
df['月收入'] = df['月收入'].fillna(df['月收入'].median())
df['家属数量'] = df['家属数量'].fillna(0)

# 保存清洗后数据
df.to_csv(os.path.join(BASE_DIR, 'data', 'cleaned_data.csv'), index=False, encoding='utf-8-sig')

# ==========================================
# 1. 用户画像 - 年龄分布
# ==========================================
print("\n=== 1. 年龄分布与违约率 ===")

df['年龄段'] = pd.cut(df['年龄'], bins=[0, 25, 35, 45, 55, 65, 100], 
                     labels=['18-25岁', '26-35岁', '36-45岁', '46-55岁', '56-65岁', '65岁以上'])

age_stats = df.groupby('年龄段', observed=True).agg(
    样本数=('是否违约', 'count'),
    违约数=('是否违约', 'sum'),
).reset_index()
age_stats['样本占比%'] = (age_stats['样本数'] / len(df) * 100).round(2)
age_stats['违约率%'] = (age_stats['违约数'] / age_stats['样本数'] * 100).round(2)
print(age_stats.to_string(index=False))

# ==========================================
# 2. 用户画像 - 收入分布
# ==========================================
print("\n=== 2. 收入分布与违约率 ===")

df['收入段'] = pd.cut(df['月收入'], bins=[0, 3000, 5000, 8000, 12000, 20000, 1000000],
                     labels=['3k以下', '3k-5k', '5k-8k', '8k-12k', '12k-20k', '20k以上'])

income_stats = df.groupby('收入段', observed=True).agg(
    样本数=('是否违约', 'count'),
    违约数=('是否违约', 'sum'),
).reset_index()
income_stats['样本占比%'] = (income_stats['样本数'] / len(df) * 100).round(2)
income_stats['违约率%'] = (income_stats['违约数'] / income_stats['样本数'] * 100).round(2)
print(income_stats.to_string(index=False))

# ==========================================
# 3. 用户画像 - 负债比率
# ==========================================
print("\n=== 3. 负债比率与违约率 ===")

df['负债比率段'] = pd.cut(df['负债比率'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 10],
                        labels=['20%以下', '20%-40%', '40%-60%', '60%-80%', '80%-100%', '100%以上'])

debt_stats = df.groupby('负债比率段', observed=True).agg(
    样本数=('是否违约', 'count'),
    违约数=('是否违约', 'sum'),
).reset_index()
debt_stats['样本占比%'] = (debt_stats['样本数'] / len(df) * 100).round(2)
debt_stats['违约率%'] = (debt_stats['违约数'] / debt_stats['样本数'] * 100).round(2)
print(debt_stats.to_string(index=False))

# ==========================================
# 4. 用户画像 - 信贷笔数
# ==========================================
print("\n=== 4. 未结清信贷笔数与违约率 ===")

df['信贷笔数段'] = pd.cut(df['未结清信贷笔数'], bins=[-0.1, 2, 5, 8, 12, 20, 100],
                        labels=['0-2笔', '3-5笔', '6-8笔', '9-12笔', '13-20笔', '20笔以上'])

loan_stats = df.groupby('信贷笔数段', observed=True).agg(
    样本数=('是否违约', 'count'),
    违约数=('是否违约', 'sum'),
).reset_index()
loan_stats['样本占比%'] = (loan_stats['样本数'] / len(df) * 100).round(2)
loan_stats['违约率%'] = (loan_stats['违约数'] / loan_stats['样本数'] * 100).round(2)
print(loan_stats.to_string(index=False))

# ==========================================
# 可视化
# ==========================================

# 图1：年龄分布与违约率
fig, ax1 = plt.subplots(figsize=(10, 6))
x = range(len(age_stats))
ax1.bar(x, age_stats['样本占比%'], color='#3498db', alpha=0.6, label='样本占比')
ax1.set_ylabel('样本占比 (%)', color='#3498db')
ax1.set_xlabel('年龄段')
ax1.set_xticks(x)
ax1.set_xticklabels(age_stats['年龄段'])

ax2 = ax1.twinx()
ax2.plot(x, age_stats['违约率%'], color='#e74c3c', marker='o', linewidth=2, label='违约率')
ax2.set_ylabel('违约率 (%)', color='#e74c3c')

plt.title('不同年龄段样本占比与违约率')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '01_age_profile.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图2：收入分布与违约率
fig, ax1 = plt.subplots(figsize=(10, 6))
x = range(len(income_stats))
ax1.bar(x, income_stats['样本占比%'], color='#2ecc71', alpha=0.6, label='样本占比')
ax1.set_ylabel('样本占比 (%)', color='#2ecc71')
ax1.set_xlabel('月收入段')
ax1.set_xticks(x)
ax1.set_xticklabels(income_stats['收入段'])

ax2 = ax1.twinx()
ax2.plot(x, income_stats['违约率%'], color='#e74c3c', marker='o', linewidth=2, label='违约率')
ax2.set_ylabel('违约率 (%)', color='#e74c3c')

plt.title('不同收入段样本占比与违约率')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '01_income_profile.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图3：负债比率与违约率
fig, ax1 = plt.subplots(figsize=(10, 6))
x = range(len(debt_stats))
ax1.bar(x, debt_stats['样本占比%'], color='#9b59b6', alpha=0.6, label='样本占比')
ax1.set_ylabel('样本占比 (%)', color='#9b59b6')
ax1.set_xlabel('负债比率')
ax1.set_xticks(x)
ax1.set_xticklabels(debt_stats['负债比率段'])

ax2 = ax1.twinx()
ax2.plot(x, debt_stats['违约率%'], color='#e74c3c', marker='o', linewidth=2, label='违约率')
ax2.set_ylabel('违约率 (%)', color='#e74c3c')

plt.title('不同负债比率样本占比与违约率')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '01_debt_profile.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图4：整体违约率饼图
plt.figure(figsize=(8, 6))
default_rate = df['是否违约'].mean() * 100
plt.pie([100-default_rate, default_rate], labels=['正常客户', '违约客户'], 
        colors=['#7bed9f', '#ff6b6b'], autopct='%1.2f%%', startangle=90)
plt.title('整体客户违约率分布')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, '01_default_rate.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n用户画像分析完成！")
