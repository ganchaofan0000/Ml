
import random
import numpy as np
import pandas as pd


def loadDataSet(path):
    """
    导入数据
    @ return dataSet: 读取的数据集
    """
    # 对数据进行处理
    reader = pd.read_csv(path)
    # 删除名称为‘编号’的的列
    reader.drop('id', axis=1, inplace=True)
    # 获得列名,列表形式
    labelSet = reader.columns.values
    # 获得列表类型的数据
    dataSet = reader.values
    return dataSet,labelSet


#knn
#距离
def distance(d1,d2):
    # 两个数据间的距离
    res=np.sum([(d1[i]-d2[i])**2 for i in range(1,d1.shape[0]-1)])
    return res**0.5


def knn(data, trainSet, k=5):
    #距离
    res=[
        {"result":train[0],"distance":distance(data,train)}
        for train in trainSet
    ]
    #排序
    res=sorted(res,key=lambda item:item['distance'])

    #取得K个
    res2=res[0:k]

    #加权平均
    result={'B':0,'M':0}

    #总距离
    sum=0
    for r in res2:
        sum+=r['distance']
    
    for r in res2:
        result[r['result']]+=1-r['distance']/sum

    if result['B']>result['M']:
        return 'B'
    else:
        return 'M'


def fit(train,test):
    correct = 0
    resultlist = []
    for test1 in test:
        result = test1[0]
        result2 = knn(test1,train)
        resultlist.append(result2)
        if result == result2:
            correct += 1
    correct = 100 * correct / len(test)
    return resultlist,correct

def main():
    # 数据集路径
    path='../DataSet/data.csv'
    # 数据集
    dataSet, labelSet=loadDataSet(path)
    # 随机化数据集
    random.shuffle(dataSet)
    # 划分数据集
    n = len(dataSet) // 3
    test_set = dataSet[0:n]
    train_set = dataSet[n:]
    result,correct = fit(train_set,test_set)
    print('结果为%s'%result)
    print("正确率为%s"%correct)

main()