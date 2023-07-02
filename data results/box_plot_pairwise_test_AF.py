"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

data = pd.read_csv('output_f1score_mean.csv')
#data = pd.read_csv('output_accuracy_mean.csv')
data.set_index('Model', inplace=True)
t_results = []
for model in data.index:
    if model == 'XGBoost':
        continue
    t_stat, p_val = ttest_rel(data.loc['XGBoost'], data.loc[model])
    t_results.append({'model': model, 't_statistic': t_stat, 'p_value': p_val})
    mean1 = data.loc['XGBoost'].mean()
    mean2 = data.loc[model].mean()
    std1 = data.loc['XGBoost'].std()
    std2 = data.loc[model].std()

    print(f"Pairwise T-test: XGBoost vs {model}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"XGBoost_mean: {mean1:.3f} {model}_mean: {mean2:.3f}")
    print(f"XGBoost_std: {std1:.3f} {model}_std: {std2:.3f}")
    print(f"p-value: {p_val:.12f}")
    print("=" * 50)

colors = ['#FFC977', '#DB5461', '#20ABD2', '#7E70AB', '#C9C8D6']

plt.figure(figsize=(3.3, 6))
ax = sns.boxplot(data=data.T, color='white', linewidth=1, fliersize=0)
for i, patch in enumerate(ax.artists):
    patch.set_edgecolor(colors[i % len(colors)])
    patch.set_alpha(1)


sns.stripplot(data=data.T, palette=colors, marker='o', facecolor='white', linewidth=0.8, alpha=1, size=4.5)
plt.title("")
plt.ylabel("F1 Score", fontsize=20)
#plt.ylabel("Accuracy", fontsize=20)
plt.xlabel("")
plt.xticks(range(len(data.index)), data.index, rotation=45)
plt.grid(which='both', linestyle='-', alpha=0.3)
plt.tick_params(axis='y', which='major', labelsize=15)
plt.tick_params(axis='x', labelsize=14)
plt.ylim(0.625, 0.875)
plt.tight_layout()
plt.savefig('f1score_ptest.png', dpi=300)
#plt.savefig('accuracy_ptest.png', dpi=300)
plt.show()
