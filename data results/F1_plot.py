"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('output_f1score_mean.csv')
data = pd.read_csv('output_f1score_std.csv')
plt.style.use('seaborn-paper')
x = list(df.columns)[1:]
x_std = list(data.columns)[1:]
colors = ['#FFC977', '#DB5461', '#20ABD2', '#7E70AB', '#C9C8D6']
#markers = ['o', 'd', 's', 'v', '^']
fig, ax = plt.subplots(figsize=(18, 6))
for i in range(5):
    y = df.iloc[i][1:]
    label = df.iloc[i][0]
    y = [1 * mean for mean in y]
    y_std = data.iloc[i][1:]
    y_std = [1.8 * std for std in y_std]
    y = np.array(y)
    y_std = np.array(y_std)
    ax.plot(x, y, color=colors[i],
            label=label, linewidth=4,
            linestyle='-', markersize=10,
            markeredgecolor='black', markeredgewidth=1) #marker=markers[i]
    ax.fill_between(x, y - y_std, y + y_std, color=colors[i], alpha=0.3)
ax.set_xlim([-1, 60])
#ax.margins(x=0.1, y=0.1)
ax.legend(fontsize=20, ncol=5, loc='upper left', bbox_to_anchor=(0, 1.2), framealpha=1)
ax.set_xlabel('Times Window(s)', fontsize=20)
ax.set_ylabel('F1-score', fontsize=20)
ax.set_ylim()
ax.tick_params(axis='y', which='major', labelsize=16)
ax.tick_params(axis='x', labelsize=15)
ax.grid(which='both', linestyle='-', alpha=0.3)
#plt.grid(which='both', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.savefig('F1_dif_model.png', dpi=400)
plt.show()

