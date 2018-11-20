import pandas as pd
import numpy as np
from math import radians, atan, tan, sin, acos, cos
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('data/train_new.csv', low_memory=False)
test = pd.read_csv('data/test_new.csv', low_memory=False)


# ################################# 准备坐标点数据################################
trL = train.shape[0] * 2
X = np.concatenate([train[['start_lat', 'start_lon']].values,
                    train[['end_lat', 'end_lon']].values,
                    test[['start_lat', 'start_lon']].values])
# #############################################################################
# 对经纬度坐标点进行密度聚类
db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(X)
labels = db.labels_

# 打印聚类数
n_clusters_ = len(set(labels))
print('Estimated number of clusters: %d' % n_clusters_)


# 训练集聚类label
info = pd.DataFrame(X[:trL,:], columns=['lat', 'lon'])
info['block_id'] = labels[:trL]
clear_info = info.loc[info.block_id != -1, :]
print('The number of miss start block in train data', (info.block_id.iloc[:trL//2] == -1).sum())
print('The number of miss end block in train data', (info.block_id.iloc[trL//2:] == -1).sum())
# 测试集聚类label
test_info = pd.DataFrame(X[trL:,:], columns=['lat', 'lon'])
test_info['block_id'] = labels[trL:]
print('The number of miss start block in test data', (test_info.block_id == -1).sum())

# 将聚类label拼接到训练集和测试集上
train['start_block'] = info.block_id.iloc[:trL//2].values
train['end_block'] = info.block_id.iloc[trL//2:].values
test['start_block'] = test_info.block_id.values
good_train_idx = (train.start_block != -1) & (train.end_block != -1)
print('The number of good training data', good_train_idx.sum())
good_train = train.loc[good_train_idx, :]
print('saving new train & test data')
good_train.to_csv('data/good_train.csv', index=None)
test.to_csv('data/good_test.csv', index=None)


# 为训练集和测试集生成is_holiday 和 hour字段
def transformer(df):
    special_holiday = ['2018-01-01'] + ['2018-02-%d' % d for d in range(15, 22)] + \
                      ['2018-04-%2d' % d for d in range(5, 8)] + \
                      ['2018-04-%d' % d for d in range(29, 31)] + ['2018-05-01'] + \
                      ['2018-06-%d' % d for d in range(16, 19)] + \
                      ['2018-09-%d' % d for d in range(22, 25)] + \
                      ['2018-10-%2d' % d for d in range(1, 8)]
    special_workday = ['2018-02-%d' % d for d in [11, 24]] + \
                      ['2018-04-08'] + ['2018-04-28'] + \
                      ['2018-09-%d' % d for d in range(29, 31)]
    for t_col in ['start_time']:
        tmp = df[t_col].map(pd.Timestamp)
        df['hour'] = tmp.map(lambda t: t.hour // 3)
        df['half'] = tmp.map(lambda t: t.minute // 30)
        df['day'] = tmp.map(lambda t: t.dayofweek)
        tmp_date = df[t_col].map(lambda s: s.split(' ')[0])
        not_spworkday_idx = ~tmp_date.isin(special_workday)
        spholiday_idx = tmp_date.isin(special_holiday)
        weekend_idx = (df['day'] >= 5)
        df['is_holiday'] = ((weekend_idx & not_spworkday_idx) | spholiday_idx).astype(int)

train = pd.read_csv('data/good_train.csv', low_memory=False)
test = pd.read_csv('data/good_test.csv', low_memory=False)
transformer(train)
transformer(test)


# 根据训练集 计算朴素贝叶斯算法需要使用的 条件概率
Probability = {}
## P(start_block|end_block)
name = 'start_block'
pname = 'P(start_block|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts()) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
print(tmp.head())
Probability[pname] = tmp
## P(out_id|end_block)
name = 'out_id'
pname = 'P(out_id|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts()) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
## P(is_holiday|end_block)
name = 'is_holiday'
pname = 'P(is_holiday|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (1.0 * g[name].value_counts() + 3.) / (len(g) + 10)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
## P((is_holiday, hour)|end_block)
pname = 'P((is_holiday, hour)|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: (5 + 1.0 * g.groupby(['is_holiday', 'hour']).size()) / (len(g))
tmp = train.groupby('end_block').apply(tmp_func).reset_index().rename(columns={0: pname})
print(tmp.head())
Probability[pname] = tmp
## P(day|end_block)
name = 'day'
pname = 'P(day|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g[name].value_counts() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
## P(hour|end_block)
name = 'hour'
pname = 'P(hour|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g[name].value_counts() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index()
tmp.columns = ['end_block', name, pname]
Probability[pname] = tmp
## P((hour,half)|end_block)
pname = 'P((hour,half)|end_block)'
print('calculating %s' % pname)
tmp_func = lambda g: 1.0 * g.groupby(['hour', 'half']).size() / len(g)
tmp = train.groupby('end_block').apply(tmp_func).reset_index().rename(columns={0: pname})
Probability[pname] = tmp


# 根据训练集 计算先验概率
pname = 'P(end_block)'
print('calculating %s' % pname)
tmp = train.end_block.value_counts().reset_index()
tmp.columns = ['end_block', pname]
Probability[pname] = tmp


## 计算后验概率
## P(end_block|(start_block, out_id, is_holiday, hour)) = P(end_block) *
##                         P(start_block|end_block) * P(out_id|end_block) * P((is_holiday, hour)|end_block)
is_local = False  # 是否线下验证
if is_local:
    predict_info = train.copy()
    predict_info = predict_info.rename(columns={'end_block': 'true_end_block', 'end_lat': 'true_end_lat', 'end_lon': 'true_end_lon'})
else:
    predict_info = test.copy()
##
predict_info = predict_info.merge(Probability['P(out_id|end_block)'], on='out_id', how='left')
print(predict_info['P(out_id|end_block)'].isnull().sum())
predict_info['P(out_id|end_block)'] = predict_info['P(out_id|end_block)'].fillna(1e-5)
##
predict_info = predict_info.merge(Probability['P(is_holiday|end_block)'], on=['is_holiday', 'end_block'], how='left')
print(predict_info['P(is_holiday|end_block)'].isnull().sum())
predict_info['P(is_holiday|end_block)'] = predict_info['P(is_holiday|end_block)'].fillna(1e-4)
##
predict_info = predict_info.merge(Probability['P(day|end_block)'], on=['day', 'end_block'], how='left')
print(predict_info['P(day|end_block)'].min(), predict_info['P(day|end_block)'].isnull().sum())
predict_info['P(day|end_block)'] = predict_info['P(day|end_block)'].fillna(1e-4)
##
predict_info = predict_info.merge(Probability['P((is_holiday, hour)|end_block)'], on=['is_holiday', 'hour', 'end_block'], how='left')
print(predict_info['P((is_holiday, hour)|end_block)'].isnull().sum())
predict_info['P((is_holiday, hour)|end_block)'] = predict_info['P((is_holiday, hour)|end_block)'].fillna(1e-4)
##
predict_info = predict_info.merge(Probability['P(start_block|end_block)'], on=['start_block', 'end_block'], how='left')
print(predict_info['P(start_block|end_block)'].isnull().sum())
predict_info['P(start_block|end_block)'] = predict_info['P(start_block|end_block)'].fillna(1e-5)
##
predict_info = predict_info.merge(Probability['P(end_block)'], on='end_block', how='left')
print(predict_info['P(end_block)'].isnull().sum())
predict_info['P(end_block)'] = predict_info['P(end_block)'].fillna(1e-1)


predict_info['P(end_block|(start_block, out_id, is_holiday, hour))'] = predict_info['P((is_holiday, hour)|end_block)'] * \
                                                                       predict_info['P(out_id|end_block)'] * \
                                                                       predict_info['P(start_block|end_block)'] * \
                                                                       predict_info['P(end_block)']
which_probability = 'P(end_block|(start_block, out_id, is_holiday, hour))'


# 生成每个聚类label的经纬度
block_lat_lon = train.groupby('end_block')[['end_lat', 'end_lon']].mean().reset_index()
predict_info = predict_info.merge(block_lat_lon, on='end_block', how='left')
print(predict_info[['start_lat', 'start_lon', 'end_lat', 'end_lon']].describe())


predict_result = predict_info.groupby('r_key').apply(lambda g: g.loc[g[which_probability].idxmax(), :]).reset_index(drop=True)
if not is_local:
    output_result = test[['r_key', 'start_lat', 'start_lon']].merge(predict_result[['r_key', 'end_lat', 'end_lon']], on='r_key', how='left')
    print(output_result.end_lat.isnull().sum())
    # 冷启动暂时用其实经纬度作为预测结果
    nan_idx = output_result.end_lat.isnull()
    output_result.loc[nan_idx, 'end_lat'] = output_result['start_lat'][nan_idx]
    output_result.loc[nan_idx, 'end_lon'] = output_result['start_lon'][nan_idx]
    #output_result[['start_lat', 'end_lat', 'end_lon']].describe()
    print(output_result.head())
    print(output_result.info())
    output_result[['r_key', 'end_lat', 'end_lon']].to_csv('result/bayes.csv', index=None)