
# ID3算法，可处理连续值（基于列表的形式）
import re
from collections import Counter
import numpy as np
import pandas as pd
import DrawTree


def loadDataSet(path):
    """
    导入数据
    @ return dataSet: 读取的数据集
    """
    # 对数据进行处理
    reader = pd.read_csv(path,delimiter=' ')
    # 删除名称为‘编号’的的列
    # reader.drop('编号', axis=1, inplace=True)
    # reader['密度'] = reader['密度'].astype('float')
    # reader['含糖率'] = reader['含糖率'].astype('float')
    # 获得列名,列表形式
    labelSet = reader.columns.values
    # 获得字典类型的数据
    dataSet = reader.values
    X=dataSet[:,:-1]
    y=dataSet[:,-1]
    return X,y,labelSet


def calcShannonEnt(y):
    """
    计算给定数据集的信息熵
    @ param dataSet: DataSet
    @ return shannonEnt: 香农熵
    """
    numEntries = len(y)
    shannonEnt = 0.0
    for key in Counter(y).most_common():
        prob = float(key[1]) / numEntries
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt


def chooseBestFeature(X,y):
    """
    求解信息增益，选择最优的划分属性,
    @ param dataSet: DataSet
    @ return bestFeature: 最佳划分属性
    """
    # D的信息增益
    baseEntroy = calcShannonEnt(y)
    # 最优信息增益的初值
    bestInfoGain = 0.0
    # 最有划分属性的初值
    bestFeature = -1
    for i in range(X.shape[1]):
        # 获取第i个特征所有可能的取值
        featureList = X[:,i]
        # 连续值的处理
        if isinstance(featureList[0],float) or isinstance(featureList[0],int):
            # 产生n-1个候选划分点
            sortfeatList = sorted(featureList)
            # n-1个候选节点
            splitList = []
            for j in range(len(sortfeatList) - 1):
                # round函数用于格式化保留到小数点后几位
                splitList.append(round((sortfeatList[j] + sortfeatList[j + 1]) / 2.0, 3))
            # 最优候选节点信息增益初值
            bestInfoGain1 = 0.0
            # 候选节点初值
            bestSplit=-1
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for value in splitList:
                newGainIndex = 0.0
                left = y[X[:, i] <= value]
                right = y[X[:, i] > value]
                prob0 = len(left) / float(len(y))
                newGainIndex += prob0 * calcShannonEnt(left)
                prob1 = len(right) / float(len(y))
                newGainIndex += prob1 * calcShannonEnt(right)
                inforGain1 = baseEntroy - newGainIndex
                if inforGain1 > bestInfoGain1:
                    bestInfoGain1 = inforGain1
                    bestSplit = value
            # 比较连续的最优节点与整个数据集上的最优属性的信息增益大小，选取较大的那个
            if bestInfoGain1 > bestInfoGain:
                bestInfoGain = bestInfoGain1
                # 如果连续值为最优划分属性，则将bestFeature标记为“feature_index<=value”的模式
                bestFeature = str(i)+'<='+str(bestSplit)
        # 离散值的处理
        else:
            # 去除重复值
            uniqueVals = set(featureList)
            newEntropy = 0.0
            for value in uniqueVals:
                sub_y = y[X[:,i] == value]
                # 特征label的数据集占总数的比例
                prob = len(sub_y) / float(len(y))
                valueEntroy = calcShannonEnt(sub_y)
                newEntropy += prob*valueEntroy
            inforGain = baseEntroy - newEntropy
            # 比较整个数据集上的最优属性的信息增益大小，选取较大的那个
            if inforGain > bestInfoGain:
                bestInfoGain = inforGain
                bestFeature = i
    return bestFeature


def createTree(X, y,labels):
    """
    构造决策树
    @ param dataSet: DataSet
    @ param labels: 标签集
    @ return myTree: 决策树
    """

    # 当dataSet中类别完全相同时停止
    if (y.tolist()).count(y[0]) == len(y.tolist()):
        return y[0]

    # labels为空时时，返回数量最多的
    if (len(labels) == 1):
        return Counter(y).most_common()[0][0]

    # 获取最佳划分属性
    bestFeat_index = chooseBestFeature(X,y)
    # # 对离散值的处理
    if isinstance(bestFeat_index,int):
        bestFeatLabel=labels[bestFeat_index]
        myTree = {bestFeatLabel: {}}
        # 清空labels[bestFeat_index]
        labels=labels[labels != bestFeatLabel]
        uniqueVals = np.unique(X[:,bestFeat_index])
        for value in uniqueVals:
            # 使的传入递归中的是labels一个复制，既地址不同，不会影响原来的labels
            labelscopy = labels[:]
            # 分支节点数据集
            X_chridren=X[X[:,bestFeat_index] == value,:]
            X_chridren=np.delete(X_chridren,bestFeat_index,axis=1)
            if(len(X_chridren)==0):
                return Counter(y).most_common()[0][0]
            # 递归调用创建决策树
            y_chridren = y[X[:,bestFeat_index] == value]
            myTree[bestFeatLabel][value] = createTree(X_chridren,y_chridren, labelscopy)
    #对连续值的处理
    else:
        bestFeat = int(re.findall('(.*?)<=', bestFeat_index)[0])
        bestPartValue = float(re.findall('<=(\d+\.\d+)', bestFeat_index)[0])
        bestFeatLabel = labels[bestFeat] + '<=' + str(bestPartValue)
        myTree = {bestFeatLabel: {}}
        subLabels = labels[:]
        left = X[:, bestFeat] <= bestPartValue
        right = X[:, bestFeat] > bestPartValue
        # 构建左子树
        valueLeft = 'Y'
        myTree[bestFeatLabel][valueLeft] = createTree(X[left,:],y[left],subLabels)
        # 构建右子树
        valueRight = 'N'
        myTree[bestFeatLabel][valueRight] = createTree(X[right,:],y[right],subLabels)
    return myTree


def main():
    path='../DataSet/watermelon3_0a_Ch.txt'
    X,y,labelSet=loadDataSet(path)
    tree = createTree(X,y,labelSet)
    DrawTree.createPlot(tree)
    print(tree)


main()