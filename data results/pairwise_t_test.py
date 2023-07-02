"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn
"""

import pandas as pd
from scipy.stats import ttest_rel
data = pd.read_csv('/path/data')
data.set_index('Model', inplace=True)
for model in data.index:
    if model == 'XGBoost':
        continue
    t_stat, p_val = ttest_rel(data.loc['XGBoost'], data.loc[model])
    mean1 = data.loc['XGBoost'].mean()
    mean2 = data.loc[model].mean()
    std1 = data.loc['XGBoost'].std()
    std2 = data.loc[model].std()
    print(f"Pairwise T-test: XGBoost vs {model}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"XGBoost_mean :{mean1:.3f} {model}_mean : {mean2:.3f}")
    print(f"XGBoost_mean :{std1:.3f} {model}_mean : {std2:.3f}")
    print(f"p-value: {p_val:.3f}")
    print("="*50)

