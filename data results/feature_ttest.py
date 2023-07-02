import pandas as pd
from scipy.stats import ttest_rel

# 读取文本文件，以制表符分隔
df = pd.read_csv('/Users/zhenghongtao/Desktop/面向自动驾驶的劝导技术应用/output/14s_f1score_plot.csv')   #14s_accuracy_ttest

# 对每一列的数据进行配对样本t检验，将结果存储在一个字典中
result = {}
for col in df.columns[1:]:
    t, p = ttest_rel(df['All'], df[col])   #Accuracy
    result[col] = {'t_statistic': t, 'p_value': p}

means = df.mean()
stds = df.std()

# 输出结果
for col in result:
    print('Column:', col)
    print('Mean:', means[col])
    print('Standard deviation:', stds[col])
    print('T-statistic:', result[col]['t_statistic'])
    print('P-value:', result[col]['p_value'])
    print()
