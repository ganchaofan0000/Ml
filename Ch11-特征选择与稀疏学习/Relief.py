import heapq

import pandas as pd
import numpy as np


def loadDataSet(path):
    """
    导入数据
    @ return dataSet: 读取的数据集
    """
    # 对数据进行处理
    reader = pd.read_csv(path)
    # 删除名称为‘编号’的的列
    reader.drop('编号', axis=1, inplace=True)
    # 获得列名,列表形式
    labelSet = reader.columns.values
    # 获得列表类型的数据
    dataSet = reader.values
    return dataSet,labelSet


class Filter:
    def __init__(self, data, label, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param t: 统计量分量阈值
        :param k: 选取的特征的个数
        """
        self.data = data
        self.label = label
        self.t = t
        self.k = k

    def distance(self,x1,x2):
        thedistance = 0
        for i in range(len(x1)-1):
            if isinstance(x1[i],int) or isinstance(x1[i],float):
                thedistance += (x1[i]-x2[i])**2
            else:
                if x1[i] != x2[i]:
                    thedistance += 1
                else:
                    thedistance += 0
        return thedistance
    # 数据标准化（将离散型数据处理成连续型数据，比如字符到数值）
    def get_data(self):
        new_data = self.data
        for i in range(new_data.shape[1]):
            col = self.data[:,i]
            if isinstance(col[0],int) or isinstance(col[0],float):
                new_data[:,i] = (new_data[:,i] - new_data[:,i].min())/(new_data[:,i].max()-new_data[:,i].min())
        return new_data

    # 返回一个样本的猜中近邻和猜错近邻
    def get_neighbors(self, row):
        data = self.get_data()
        row_type = row[-1]
        right_data = data[data[:, -1] == row_type]
        wrong_data = data[data[:, -1] != row_type]
        right_distance = []
        for a in right_data:
            right_distance.append(self.distance(row, a))
        wrong_distance = []
        for b in wrong_data:
            wrong_distance.append(self.distance(row,b))
        right_distance.remove(min(right_distance))
        return np.argmin(right_distance),np.argmin(wrong_distance)



    # 过滤式特征选择
    def relief(self):
        sample = self.get_data()
        # y_class = set(sample[:, -1])
        score = []
        # 遍历所有属性
        ad=sample.shape[1]-1
        for j in range(sample.shape[1]-1):
            j_score = 0
            # 遍历所有样本
            for row in sample:    # 采样次数
                # one_score = dict()
                NearHit, NearMiss = self.get_neighbors(row)
                # 遍历不同的类
                if isinstance(row[j],int) or isinstance(row[j],float):
                    diff1 = abs(row[j]-sample[NearHit][j])
                else:
                    if row[j] == sample[NearHit][j]:
                        diff1 = 0
                    else:
                        diff1 = 1
                if isinstance(row[j],int) or isinstance(row[j],float):
                    diff2 = abs(row[j]-sample[NearMiss][j])
                else:
                    if row[j] == sample[NearMiss][j]:
                        diff2 = 0
                    else:
                        diff2 = 1
                j_score += -diff1**2 + diff2**2
            score.append(j_score)
        return score

    # 返回最终选取的特征
    def get_final(self):
        score = self.relief()
        print(score)
        label = self.label
        # 最大的k个数的索引
        max_num_index_list = map(score.index, heapq.nlargest(self.k, score))
        bestlabel = {label[i]:score[i] for i in list(max_num_index_list)}
        return bestlabel


if __name__ == '__main__':
    path = '../DataSet/watermelon_3.csv'
    data,label = loadDataSet(path)
    f = Filter(data, label, 0.8, 6)
    bestlabel = f.get_final()
    print(bestlabel)