# 汽车目的地智能预测大赛
比赛链接：http://www.pkbigdata.com/common/cmpt/%E6%B1%BD%E8%BD%A6%E7%9B%AE%E7%9A%84%E5%9C%B0%E6%99%BA%E8%83%BD%E9%A2%84%E6%B5%8B%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

MultinomialNB 0.45
svm 0.7
增加特征logistic 0.45
增加特征xgb 0.423
朴素贝叶斯0.427

朴素贝叶斯
数据清洗：
使用DBSCAN对所有数据的起点与终点进行密度聚类，结果作为数据的标签，只保留训练集中不被判定为噪声的数据。

特征工程：
  根据日期为每个数据增加：出发日（day of week）、小时、是否节假日，这几组特征。
模型：
  假设特征之间相互独立，使用朴素贝叶斯。start_block和end_block是在通过经纬度聚类出来的一个地点类别标签，等于说是把回归问题变成了分类问题。

计算条件概率（似然度）——
P(start_block|end_block)、P(out_id|end_block)、P(is_holiday|end_block)、P((is_holiday, hour)|end_block)、P(day|end_block)、P(hour|end_block)、P((hour,half)|end_block)

计算先验概率——
	P(end_block)

使用merge，分别以——
['out_id']，[['is_holiday', 'end_block']，['is_holiday', 'hour', 'end_block']，['start_block', 'end_block']，['end_block']为键左外连接，为测试数据增加上述条件概率列。

计算测试集的P(end_block|(start_block, out_id, is_holiday, hour))，取最大值作为end_block的预测值。



Xgboost
这个版本的代码，在方案上和朴素贝叶斯有了不同。
主要思路：讲训练集的经纬度进行geohash编码。专门针对不同的车辆训练不同的模型。单独对一辆车的所有记录作为训练集训练模型，然后对该辆车做出预测。
代码执行思路；
提取第i辆车的所有数据，将所有去过的地点作为一个多分类标签，在xgboost模型下进行训练。最后输入测试集中该第i辆车的特征，由模型给出预测的多分类标签。

注：训练集和测试集中涵盖了5033辆车。对每一辆车的数据单独训练，设置最大树深度为12，每个模型迭代2000次，线上评分0.4233。

