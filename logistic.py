# -*- coding: utf-8 -*-
import datetime
import os
import time
from collections import Counter
import geohash
import xgboost as xgb
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import string
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model.logistic import LogisticRegression
"""
汽车目的地智能预测大赛_knn
"""

def datetime_to_period(date_str):
    """
    描述：把时间分为24段
    返回：0到23
    """
    time_part = date_str.split(" ")[1]  # 获取时间部分
    hour_part = int(time_part.split(":")[0])  # 获取小时
    return hour_part


def date_to_period(date_str):
    """
    描述：把日期转化为对应的工作日或者节假日
    返回：0:工作日 1：节假日 2:小长假
    """
    holiday_list = ['2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18', '2018-02-19',
                    '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06', '2018-04-07', '2018-04-29',
                    '2018-04-30', '2018-05-01', '2018-06-16', '2018-06-17', '2018-06-18']  # 小长假
    switch_workday_list = ['2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28']  # 小长假补班
    workday_list = ['1', '2', '3', '4', '5']  # 周一到周五
    weekday_list = ['0', '6']  # 周六、周日，其中0表示周日
    date = date_str.split(" ")[0]  # 获取日期部分
    whatday = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime("%w")  # 把日期转化为星期
    if date in holiday_list:
        return 2
    elif date in switch_workday_list:
        return 0
    elif whatday in workday_list:
        return 0
    elif whatday in weekday_list:
        return 1


time_start = time.asctime(time.localtime(time.time()))  # 程序开始时间




"""
增加特征：
    增加了：hour ，half， day， is_holiday 四组特征
"""
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
        tmp_date = df[t_col].map(lambda s: s.split(' ')[0])     #按' '分割，取第一块元素。即2018-02-12 17:40:51取2018-02-12
        not_spworkday_idx = ~tmp_date.isin(special_workday)     #若该条数据不是special_workday，bool变量置为ture
        spholiday_idx = tmp_date.isin(special_holiday)          #若该条数据是special_holiday，bool变量置为ture
        weekend_idx = (df['day'] >= 5)                          #若该条数据是周末，bool变量置为ture
        df['is_holiday'] = ((weekend_idx & not_spworkday_idx) | spholiday_idx).astype(int)


train_data = pd.read_csv("data/train_new.csv", low_memory=False)
test_data = pd.read_csv("data/test_new.csv", low_memory=False)
transformer(train_data)
transformer(test_data)
train_data.to_csv('data/train_feature.csv', index=None)
test_data.to_csv('data/test_feature.csv', index=None)

train_data = pd.read_csv("data/train_feature.csv", low_memory=False)
test_data = pd.read_csv("data/test_feature.csv", low_memory=False)
n = 0
#统计车辆个数 5033辆车
test_out_id = Counter(test_data['out_id'])
# print(len(test_out_id))
id =0
for out_id in test_out_id.keys():
    # ----train_new数据补充字段 begin----
    train = train_data[train_data['out_id'] == out_id]  # 选择出同一个out_id的数据
    train = shuffle(train)  # 打乱顺序
    train['start_code'] = None  # 开始位置的geohash编码
    train['end_code'] = None  # 结束位置的geohash编码

    train['period'] = None  # 时间段编码（0-23）
    train['week_code'] = None  # 工作日和休息日编码
    # columns = train.columns.values.tolist()         #获取列名列表
    # print('train',columns)
#     # train ['r_key', 'out_id', 'start_time', 'end_time', 'start_lat', 'start_lon', \
#              'end_lat', 'end_lon', 'hour', 'half', 'day', 'is_holiday', 'start_code', 'end_code', 'period', 'week_code']
    for i in range(len(train)):
        train.iloc[i, 12] = geohash.encode(train.iloc[i, 4], train.iloc[i, 5], 8)  # 开始位置geohash编码 start_code
        train.iloc[i, 13] = geohash.encode(train.iloc[i, 6], train.iloc[i, 7], 8)  # 结束位置geohash编码 end_code
        train.iloc[i, 14] = datetime_to_period(train.iloc[i, 2])  # 添加时间段 period
        train.iloc[i, 15] = date_to_period(train.iloc[i, 2])  # 添加工作日、休息日编码 week_code

    #构造该id对应的整形标签
    train['end_code_int'] = None  # 结束位置的geohash编码对应的整形标签
    # columns = train.columns.values.tolist()         #获取列名列表
    # print('train',columns)
    train = train.sort_values('end_code', ascending=False)
    for i in range(len(train)):
        if i==0:
            train.iloc[i, 16] = 0
        elif train.iloc[i, 13]==train.iloc[i-1, 13]:
            train.iloc[i, 16] = train.iloc[i-1, 16]
        elif train.iloc[i, 13]!=train.iloc[i-1, 13]:
            train.iloc[i, 16] = train.iloc[i-1, 16] + 1

#     # print(train.iloc[len(train)-1, 16]) # 这个数字代表0 - train.iloc[len(train)-1 是分类标签
    num_class = train.iloc[len(train)-1, 16]+1
    # print(num_class)
#     # train.to_csv("train_freat.csv", encoding='utf-8', index=False,header=True)


    # ----train_new数据补充字段 end----

    # ----test_new数据补充字段 begin----
    test = test_data[test_data['out_id'] == out_id]
    test = shuffle(test)  # 打乱顺序
    test['period'] = None
    test['week_code'] = None
    test['start_code'] = None
    test['end_code_int'] = None
    # columns = test.columns.values.tolist()         #获取列名列表
    # print('test',columns)
#     # test ['r_key', 'out_id', 'start_time', 'start_lat', 'start_lon', 'hour', 'half', 'day', 'is_holiday',\
#              'period', 'week_code', 'start_code', 'end_code_int']
    for i in range(len(test)):
        test.iloc[i, 9] = datetime_to_period(test.iloc[i, 2])  # 添加时间段 period
        test.iloc[i, 10] = date_to_period(test.iloc[i, 2])  # 添加工作日、休息日编码 week_code
        test.iloc[i, 11] = geohash.encode(test.iloc[i, 3], test.iloc[i, 4], 8)  # 开始位置geohash编码 start_code
    # ----test_new数据补充字段 end----

    """
    训练
    """
    train_x = train[['start_lat', 'start_lon', 'period', 'week_code', 'hour', 'half', 'day', 'is_holiday']]
    train_y = train['end_code_int']
    test_x = test[['start_lat', 'start_lon', 'period', 'week_code', 'hour', 'half', 'day', 'is_holiday']]
    classifier = LogisticRegression()
    classifier.fit(train_x,train_y.astype('float'))
    predictions = classifier.predict(test_x)
    test['end_code_int'] = predictions
    # 将每一个id的end_code_int映射回end_code
    int_str = train[['out_id','end_code_int','end_code']]
    test = pd.merge(test,int_str,on=['out_id','end_code_int'],how='left')
    id = id+1
    print(id)
    # print(test['predict_int'])
    #to_csv的mode参数
    # r ：只读
    # r+ : 读写
    # w ： 新建（会对原有文件进行覆盖）
    # a ： 追加n
    # b ： 二进制文件
    if n == 0:
        test.to_csv("predict_int_logistic.csv", mode='a', encoding='utf-8', index=False,header=True)
    else:
        test.to_csv("predict_int_logistic.csv", mode='a', encoding='utf-8', index=False,header=False)

    if n % 500 == 0:
        print("已运行：" + str(n) + " " + time.asctime(time.localtime(time.time())))
    n = n + 1

print("输出结果：\n")
df = pd.read_csv("predict_int_logistic.csv")  # 预测结果文件
df.drop_duplicates(inplace=True)
print(df.shape)
df['end_lat'] = None
df['end_lon'] = None
columns = df.columns.values.tolist()         #获取列名列表
print('df',columns)
for i in range(len(df)):
    site = geohash.decode(df.iloc[i, 13])
    df.iloc[i, 14] = site[0]  # 预测横坐标
    df.iloc[i, 15] = site[1]  # 预测纵坐标
    if i % 5000 == 0:
        print("已运行" + str(i))
df = df[['r_key', 'end_lat', 'end_lon']]
df.to_csv("predict_decode_logistic.csv", encoding='utf-8', index=False)

print('\r程序运行开始时间：', time_start)
print('\r程序运行结束时间：', time.asctime(time.localtime(time.time())))
