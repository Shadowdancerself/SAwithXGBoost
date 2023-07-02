import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 创建数据框
data = pd.read_csv('/Users/zhenghongtao/Desktop/面向自动驾驶的劝导技术应用/output/14s_accuracy_plot.csv')

formula = 'Accuracy ~ C(feature) + C(random) + C(feature):C(random)'
model = ols(formula, data=data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)

print(aov_table)

tukey_results = pairwise_tukeyhsd(data['Accuracy'], data['feature'])
print(tukey_results.summary())
