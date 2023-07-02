import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 读取数据
data = pd.read_csv('/Users/zhenghongtao/Desktop/面向自动驾驶的劝导技术应用/output/output_f1score.csv')

# 进行双因素方差分析
formula = 'F1Score ~ C(times) + C(model) + C(times):C(model)'
model = ols(formula, data=data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)

# 打印方差分析结果
print(aov_table)

# 进行Tukey事后比较
tukey_results = pairwise_tukeyhsd(data['F1Score'], data['model'])

# 打印比较结果
print(tukey_results.summary())

