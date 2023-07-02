"""
使用python帮我完成一个双因素方差分析：
读取制定位置的csv；
指定因素1：为第一列
指定因素2: 为第二列
作用结果为第三列
分析因素1和因素2对作用结果的影响
"""


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np

# 读取CSV文件
df = pd.read_csv('/Users/zhenghongtao/Desktop/面向自动驾驶的劝导技术应用/output/4s_anova.csv')



# 指定因素1、因素2和作用结果的列名
factor1 = 'times'
factor2 = 'subsets'
outcome = 'Accuracy'

# 执行双因素方差分析
formula = f"{outcome} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
model = ols(formula, data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)

# 打印方差分析结果
print(aov_table)
