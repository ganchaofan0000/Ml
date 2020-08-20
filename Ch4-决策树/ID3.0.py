
import numpy as np
import pandas as pd
import DrawTree


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
    labelSet = list(reader.columns.values)
    # 获得字典类型的数据
    dataSet = reader.to_dict(orient='records')
    return dataSet,labelSet


def calcShannonEnt(dataSet,labels):
    """
    计算给定数据集的信息熵
    @ param dataSet: DataSet
    @ return shannonEnt: 香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 当前样本类型
        currentLabel = featVec[labels[-1]]
        # 如果当前类别不在labelCounts里面，则创建
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt


def splitDataSet(dataSet, label, value):
    """
    划分数据集, 提取所有满足一个特征的数据集
    @ param dataSet: DataSet
    @ param axis: 划分数据集的特征
    @ param value: 提取出来满足某特征的list
    """
    retDataSet = []
    for featVec in dataSet:
        # 将相同数据特征的提取出来
        if featVec[label] == value:
            retDataSet.append(featVec)
    return retDataSet


def chooseBestFeature(dataSet,labels):
    """
    求解信息增益，选择最优的划分属性,
    @ param dataSet: DataSet
    @ return bestFeature: 最佳划分属性
    """
    baseEntroy = calcShannonEnt(dataSet,labels)
    bestInfoGain = 0.0
    bestFeature = -1
    for label in labels[:len(labels)-1]:
        # 获取第i个特征所有可能的取值
        featureList = [example[label] for example in dataSet]
        # 去除重复值
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, label, value)
            # 特征label的数据集占总数的比例
            prob = len(subDataSet) / float(len(dataSet))
            valueEntroy = calcShannonEnt(subDataSet, labels)
            newEntropy += prob*valueEntroy
        inforGain = baseEntroy - newEntropy

        if inforGain > bestInfoGain:
            bestInfoGain = inforGain
            bestFeature = label
    return bestFeature


def majorityCnt(dataSet,labels):
    """
    返回出现次数最多的类别
    @ param classList: 类别列表
    @ return sortedClassCount[0][0]: 出现次数最多的类别
    """
    # 获得当前数据集的类别集合
    classList = [example[labels[-1]] for example in dataSet]

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=lambda item:item[1], reverse=True)
    # 返回出现次数最多的
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    构造决策树
    @ param dataSet: DataSet
    @ param labels: 标签集
    @ return myTree: 决策树
    """
    # 获得当前数据集的类别集合
    classList = [example[labels[-1]] for example in dataSet]

    # 当dataSet中类别完全相同时停止
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # labels为空时时，返回数量最多的
    if (len(labels) == 1):
        return majorityCnt(dataSet,labels)

    # 获取最佳划分属性
    bestFeatLabel = chooseBestFeature(dataSet,labels)
    myTree = {bestFeatLabel: {}}
    # 清空labels[bestFeat]
    labels.remove(bestFeatLabel)
    featValues = [example[bestFeatLabel] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 使的传入递归中的是labels一个复制，既地址不同，不会影响原来的labels
        labelscopy = labels[:]
        # 分支节点数据集
        dataSetchridren=splitDataSet(dataSet, bestFeatLabel, value)
        if(len(dataSetchridren)==0):
            return majorityCnt(dataSet,labelscopy)
        # 递归调用创建决策树
        myTree[bestFeatLabel][value] = createTree(dataSetchridren, labelscopy)
    return myTree


def main():
    path='../DataSet/西瓜数据集2.txt'
    dataSet,labelSet=loadDataSet(path)
    tree = createTree(dataSet, labelSet)
    DrawTree.createPlot(tree)
    print(tree)


main()