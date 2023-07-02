"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn

python: Feature Accumulation plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path = '/path/data'
data = pd.read_csv(path)

x_labels = data.columns.values[0:]
means = data.mean(axis=0)[0:]
stds = data.std(axis=0)[0:]
yerr = [4 * std for std in stds]
plt.style.use('seaborn-paper')


fig, ax = plt.subplots()
#ax.plot(range(len(x_labels)), means, '-', color='#1F77B4', linewidth=7, alpha=0.2)
ax.plot(range(len(x_labels)), means, marker='s', color='#1F77B4', markeredgecolor='black', markeredgewidth=1, markersize=7, label='F1-score')
#ax.errorbar(range(len(x_labels)), means, yerr=yerr, fmt='s', color='#1F77B4', markeredgecolor='black', markeredgewidth=1, capsize=3, ecolor='black', elinewidth=1, markersize=7)
ax.fill_between(range(len(x_labels)), np.subtract(means, yerr), np.add(means, yerr), color='#1F77B4', alpha=0.2)


path = '/path/data'
data = pd.read_csv(path)
x_labels = data.columns.values[0:]
means = data.mean(axis=0)[0:]
stds = data.std(axis=0)[0:]
yerr = [4 * std for std in stds]
ax.plot(range(len(x_labels)), means, marker='o', color='#FF7F0E', markeredgecolor='black', markeredgewidth=1, markersize=7, label='Accuracy')
#ax.errorbar(range(len(x_labels)), means, yerr=yerr, fmt='o', color='#FF7F0E', markeredgecolor='black', markeredgewidth=1, capsize=3, ecolor='black', elinewidth=1, markersize=7, label='Accuracy')
ax.fill_between(range(len(x_labels)), np.subtract(means, yerr), np.add(means, yerr), color='#FF7F0E', alpha=0.2)


fig.set_size_inches(14, 7)
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, ha='center')
ax.legend(loc='lower right', fontsize=20, framealpha=1)
ax.set_xlabel("Sequential feature accumulation in XGBoost performance", fontsize=20)
ax.set_ylabel("Accuracy/F1-score", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)


plt.savefig('/path/data', dpi=300, bbox_inches='tight')


