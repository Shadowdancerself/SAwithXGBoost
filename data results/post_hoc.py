import pandas as pd

# 读取csv文件
df = pd.read_csv('/Users/zhenghongtao/Desktop/面向自动驾驶的劝导技术应用/output/output_accuracy.csv')

# 计算times列中，每个值对应的model值下的Accuracy平均值 F1Score
means = df.groupby(['model', 'times'])['Accuracy'].mean().reset_index()

# 使用pivot_table方法将数据重新构建为以model为列，times为行的形式
pivot_df = means.pivot_table(index='times', columns='model', values='Accuracy')

df = pivot_df.transpose()

# 保存到指定位置
df.to_csv('/Users/zhenghongtao/Desktop/面向自动驾驶的劝导技术应用/output/output_accuracy_mean.csv')
