
# 基于ID3的后剪枝
import copy
import re

import numpy as np
import pandas as pd
import csv
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

def splitContinuousDataSet(dataSet, label, value):
    # 小于value的数据
    left = []
    # 大于value的数据
    right = []
    for featVec in dataSet:
        if featVec[label] > value:
            right.append(featVec)
        if featVec[label] <= value:
            left.append(featVec)
    return left,right


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

# 由于在Tree中，连续值特征的名称以及改为了  feature<=value的形式
# 因此对于这类特征，需要利用正则表达式进行分割，获得特征名以及分割阈值
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    if '<=' in firstStr:
        featvalue = float(re.findall('<=(\d+\.\d+)', firstStr)[0])
        featkey = re.findall('(.*?)<=', firstStr)[0]
        secondDict = inputTree[firstStr]
        #featIndex = featLabels.index(featkey)
        if testVec[featkey] <= featvalue:
            judge = 'Y'
        else:
            judge = 'N'
        for key in secondDict.keys():
            if judge == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    else:
        secondDict = inputTree[firstStr]
        #featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[firstStr] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
    return classLabel


def testing(myTree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        if classify(myTree, labels, data_test[i]) != data_test[i][labels[-1]]:
            error += 1
    # print
    # 'myTree %d' % error
    return float(error)


def testingMajor(major, data_test,labels):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][labels[-1]]:
            error += 1
    # print
    # 'major %d' % error
    return float(error)


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

# 后剪枝
def postPruningTree(inputTree, dataSet, data_test, labels):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    if '<=' in firstStr:  # 对连续的特征值，使用正则表达式获得特征标签和value
        featvalue = float(re.findall('<=(\d+\.\d+)', firstStr)[0])
        featkey = re.findall('(.*?)<=', firstStr)[0]
        left, right = splitContinuousDataSet(dataSet, featkey, featvalue)
        left_test, right_test = splitContinuousDataSet(data_test, featkey, featvalue)
        for key in secondDict.keys():  # 对每个分支
            if type(secondDict[key]).__name__ == 'dict':
                if key == 'Y':
                    inputTree[firstStr][key] = postPruningTree(secondDict[key],left,left_test,copy.deepcopy(labels))
                else:
                    inputTree[firstStr][key] = postPruningTree(secondDict[key], right, right_test, copy.deepcopy(labels))
    else: # 对离散值的处理
        for key in secondDict.keys():  # 对每个分支
            if type(secondDict[key]).__name__ == 'dict':
                dataSetchridren = splitDataSet(dataSet, firstStr, key)
                data_testchridren = splitDataSet(data_test, firstStr, key)
                inputTree[firstStr][key]=postPruningTree(secondDict[key],dataSetchridren,data_testchridren,copy.deepcopy(labels))

    if testing(inputTree,data_test, labels) <= testingMajor(majorityCnt(dataSet, labels),data_test,labels):
        return inputTree
    return majorityCnt(dataSet,labels)


def main():
    # path='DataSet/西瓜数据集2.txt'
    # dataSet,labelSet=loadDataSet(path)
    # tree = createTree(dataSet, labelSet)
    # DrawTree.createPlot(tree)
    # print(tree)
    path1 = '../DataSet/西瓜1'
    path2 = '../DataSet/西瓜2'
    data_test, labelSet = loadDataSet(path2)
    dataSet, labelSet = loadDataSet(path1)
    tree = createTree(dataSet, labelSet)
    DrawTree.createPlot(tree)
    tree=postPruningTree(tree,dataSet,data_test,labelSet)
    DrawTree.createPlot(tree)

main()