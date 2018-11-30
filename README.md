# 汽车目的地智能预测大赛
比赛链接：http://www.pkbigdata.com/common/cmpt/%E6%B1%BD%E8%BD%A6%E7%9B%AE%E7%9A%84%E5%9C%B0%E6%99%BA%E8%83%BD%E9%A2%84%E6%B5%8B%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

bayes.py-朴素贝叶斯baseline-0.42

1.使用DBSCAN对所有数据的起点与终点进行密度聚类，结果作为数据的标签，只保留训练集中不被判定为噪声的数据。

2.根据日期为每个数据增加：出发日（day of week）、小时、是否节假日

3.计算条件概率（似然度）——

	P(start_block|end_block)、P(out_id|end_block)、P(is_holiday|end_block)、
	P((is_holiday, hour)|end_block)、P(day|end_block)、
	P(hour|end_block)、P((hour,half)|end_block)

4.计算先验概率——
	P(end_block)
	
5.使用merge，分别以——
	['out_id']，[['is_holiday', 'end_block']，['is_holiday', 'hour', 'end_block']，
	['start_block', 'end_block']，['end_block']
        为键左外连接，为测试数据增加上述条件概率列，具体操作见代码***（上述合并会出现同一个样本id有不同条件概率的情况，如一个end_block中有多个     out_id，做合并后一个out_id会对应多个end_block和P(out_id|end_block)）***

6.计算测试集的P(end_block|(start_block, out_id, is_holiday, hour))，取最大值作为end_block的预测值


----------------------------------------------------------------------------------------------------------------------


以下这几个版本特征工程和数据处理相同，只是分类器不同
KNN.py 0.48
logistic.py 0.48
MultinomialNB.py 0.49
svm.py 0.64
XGB.py 0.6

大致思路：对坐标进行geohasn编码，针对每一辆车训练一个模型，进行预测。
