import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = {'feature': ['scan_pattern', 'Gaze_dispersion', 'on_road', 'Total_duration_of_fixations.n-aoi',
                    'Total_duration_of_fixations.ndt', 'Total_duration_of_fixations.road',
                    'Number_of_fixation_starts.n-aoi', 'Number_of_fixation_starts.ndt',
                    'Number_of_fixation_starts.road', 'Average_pupil_diameter.n-aoi',
                    'Average_pupil_diameter.ndt', 'Average_pupil_diameter.road', 'pupil_sd.n-aoi',
                    'pupil_sd.ndt', 'pupil_sd.road', 'saccade.n-aoi', 'saccade.ndt', 'saccade.road',
                    'Blink', 'speed', 'throttle', 'steer', 'brake', 'lead_time', 'weather', 'scenario_type',
                    'density'],
        'feature_importance': [0.200358, 0.306861, 0.500914, 0.09003, 0.086132,
                               0.68911, 0.10003, 0.101353, 0.302494, 0.090033,
                               0.415179, 0.570812, 0.040054, 0.073551, 0.107772,
                               0.09003, 0.009049, 0.300573, 0.072608, 0.483596,
                               0.20747, 0.175326, 0.086685, 0.425281, 0.391114, 0.362572, 0.297578],
        'group': ['Eye tracking', 'Eye tracking', 'Eye tracking', 'Eye tracking',
                  'Eye tracking', 'Eye tracking', 'Eye tracking', 'Eye tracking',
                  'Eye tracking', 'Eye tracking', 'Eye tracking', 'Eye tracking',
                  'Eye tracking', 'Eye tracking', 'Eye tracking', 'Eye tracking',
                  'Eye tracking', 'Eye tracking', 'Eye tracking', 'Autonomous car',
                  'Autonomous car', 'Autonomous car', 'Autonomous car', 'Environment',
                  'Environment', 'Environment', 'Environment']}

df = pd.DataFrame(data)

# 按照分组的方式进行可视化
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(8, 6))
#sns.barplot(x='feature', y='feature_importance', hue='group', data=df, ax=ax, alpha=0.8, edgecolor='k', linewidth=0.5, dodge=0.2, width=0.7)
sns.barplot(x='feature', y='feature_importance', hue='group', data=df, ax=ax,
            saturation=1, alpha=0.7, edgecolor='black',
            palette=['#ED553B', '#20639B', '#3CAEA3'], dodge=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
ax.yaxis.grid(alpha=0.5)  # 设置纵坐标方格透明度为0.5
ax.set_xlabel('')
ax.set_ylabel('Feature importance', fontsize=20)
ax.legend(fontsize=24)

# 调整柱子间的间距
n_groups = len(df['feature'].unique())
#plt.setp(ax.artists, width=0.4)
plt.setp(ax.containers, width=0.8)
ax.legend(ncol=n_groups, loc='upper right')

sns.despine()
plt.tight_layout()
plt.savefig('/Users/zhenghongtao/Desktop/面向自动驾驶的劝导技术应用/output/feature_imp.png', dpi=400)
