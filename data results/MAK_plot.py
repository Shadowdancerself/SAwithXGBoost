"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn
"""

import pandas as pd
import matplotlib.pyplot as plt

colors = ['#FFC977', '#DB5461', '#20ABD2', '#7E70AB', '#C9C8D6']
markers = ['o', 's', 'd', 'v', '^']
csv_directory = ""  #
mean_file_path = csv_directory + "mean.csv"
mean_data = pd.read_csv(mean_file_path)
x_labels_mean = mean_data.columns[1:]
mean_values = mean_data.iloc[:, 1:].astype(float)
std_file_path = csv_directory + "std.csv"
std_data = pd.read_csv(std_file_path)
std_values = std_data.iloc[:, 1:].astype(float)
errorbar_scale = 4
errorbar_kwargs = {'capsize': 5, 'elinewidth': 1.5}
for i in range(mean_values.shape[0]):
    y_err = std_values.iloc[i] * errorbar_scale
    plt.errorbar(x_labels_mean, mean_values.iloc[i], yerr=y_err, label=mean_data['model'][i], marker=markers[i], color=colors[i], markeredgewidth=3,
                 **errorbar_kwargs)
plt.xlabel('Different Models', fontsize=20)
plt.ylabel('Result', fontsize=20)
plt.grid(which='major', axis='y', linestyle='-', alpha=0.3)
plt.legend(loc='best', fontsize=18, framealpha=1)
plt.tick_params(axis='both', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.savefig(csv_directory + "plot.png", dpi=300)
plt.show()

