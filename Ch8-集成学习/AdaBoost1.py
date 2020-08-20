
# ID3算法，可处理连续值
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def loadDataSet(path):
    """
    导入数据
    @ return dataSet: 读取的数据集
    """
    # 对数据进行处理
    reader = pd.read_csv(path,delimiter=' ')
    # 获得列名,列表形式
    labelSet = reader.columns.values
    # 获得字典类型的数据
    dataSet = reader.values
    X=dataSet[:,:-1]
    y=dataSet[:,-1]
    return X,y,labelSet


def gini(y, D):
    '''
    计算样本集y下的加权基尼指数
    :param y: 数据样本标签
    :param D: 样本权重
    :return:  加权后的基尼指数
    '''
    unique_class = np.unique(y)
    total_weight = np.sum(D)

    gini = 1
    for c in unique_class:
        gini -= (np.sum(D[y == c]) / total_weight) ** 2

    return gini


def chooseBestFeature(X,y,D):
    """
    求解基尼指数，选择最优的划分属性,
    @ param dataSet: DataSet
    @ return bestFeature: 最佳划分属性
    """

    total_weight = np.sum(D)
    # D的基尼值
    baseEntroy = gini(y,D)
    # 最优基尼指数的初值
    min_gini = 100.0
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
            min_gini1 = 100.0
            # 候选节点初值
            bestSplit=-1
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for value in splitList:
                y_left = y[X[:, i] <= value]
                y_right = y[X[:, i] > value]
                D_left = D[X[:, i] <= value]
                D_right = D[X[:, i] > value]
                gini_tmp = (np.sum(D_left) * gini(y_left, D_left) + np.sum(D_right) * gini(y_right, D_right)) / total_weight
                if gini_tmp < min_gini1:
                    min_gini1 = gini_tmp
                    bestSplit = value
            # 比较连续的最优节点与整个数据集上的最优属性的信息增益大小，选取较大的那个
            if min_gini1 < min_gini:
                min_gini = min_gini1
                # 如果连续值为最优划分属性，则将bestFeature标记为“feature_index<=value”的模式
                bestFeature = str(i)+'<='+str(bestSplit)
        # 离散值的处理
        else:
            # 去除重复值
            uniqueVals = set(featureList)
            gini_tmp=0.0
            for value in uniqueVals:
                y_chridren = y[X[:,i] == value]
                D_chridren = D[X[:,i] == value]
                gini_tmp += np.sum(D_chridren) * gini(y_chridren,D_chridren) / total_weight
            # 比较整个数据集上的最优属性的信息增益大小，选取较大的那个
            if gini_tmp < min_gini:
                min_gini = gini_tmp
                bestFeature = i
    return bestFeature


def predictSingle(inputTree, testVec,labels):
    firstStr = list(inputTree.keys())[0]
    classLabel = None
    if '<=' in firstStr:
        featvalue = float(re.findall('<=(\d+\.\d+)', firstStr)[0])
        featkey = re.findall('(.*?)<=', firstStr)[0]
        secondDict = inputTree[firstStr]
        index=np.argwhere(labels == featkey)[0][0]
        #featIndex = featLabels.index(featkey)
        if testVec[index] <= featvalue:
            judge = 'Y'
        else:
            judge = 'N'
        for key in secondDict.keys():
            if judge == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = predictSingle(secondDict[key], testVec,labels)
                else:
                    classLabel = secondDict[key]
    else:
        index = np.argwhere(labels == firstStr)[0][0]
        secondDict = inputTree[firstStr]
        for key in secondDict.keys():
            if testVec[index] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = predictSingle(secondDict[key], testVec,labels)
                else:
                    classLabel = secondDict[key]
    return classLabel


def predictBase(tree, X,labels):
    '''
    基于基学习器预测所有样本
    :param tree:
    :param X:
    :return:
    '''
    result = []

    for i in range(X.shape[0]):
        result.append(predictSingle(tree, X[i, :],labels))
    i=0
    return np.array(result)


def createTree(X, y,D,labels):
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
        if y.dot(D) >= 0:
            return 1
        else:
            return -1

    # 获取最佳划分属性
    bestFeat_index = chooseBestFeature(X,y,D)
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
                if y.dot(D) >= 0:
                    return 1
                else:
                    return -1
            # 递归调用创建决策树
            y_chridren = y[X[:,bestFeat_index] == value]
            D_chridren = D[X[:,bestFeat_index] == value]
            myTree[bestFeatLabel][value] = createTree(X_chridren,y_chridren,D_chridren,labelscopy)
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
        myTree[bestFeatLabel][valueLeft] = createTree(X[left,:],y[left],D[left],subLabels)
        # 构建右子树
        valueRight = 'N'
        myTree[bestFeatLabel][valueRight] = createTree(X[right,:],y[right],D[right],subLabels)
    return myTree


def adaBoostTrain(X, y, labels,tree_num=20):
    '''
    以CART决策树作为基学习器，训练adaBoost
    :param X:
    :param y:
    :param tree_num:
    :return:
    '''
    D = np.ones(y.shape) / y.shape  # 初始化权重

    trees = []  # 所有基学习器
    a = []  # 基学习器对应权重

    # agg_est = np.zeros(y.shape)

    for _ in range(tree_num):
        tree = createTree(X, y, D,labels)
        hx = predictBase(tree, X,labels)
        err_rate = np.sum(D[hx != y])

        at = np.log((1 - err_rate) / max(err_rate, 1e-16)) / 2

        # agg_est += at * hx
        trees.append(tree)
        a.append(at)

        if (err_rate > 0.5) | (err_rate == 0):  # 错误率大于0.5 或者 错误率为0时，则直接停止
            break

        # 更新每个样本权重
        err_index = np.ones(y.shape)
        err_index[hx == y] = -1

        D = D * np.exp(err_index * at)
        D = D / np.sum(D)

    return trees, a


def adaBoostPredict(X, trees, a,labels):
    agg_est = np.zeros((X.shape[0],))

    for tree, am in zip(trees, a):
        result1 = predictBase(tree, X,labels)
        agg_est += am * result1

    result = np.ones((X.shape[0],))

    result[agg_est < 0] = -1

    return result.astype(int)


def pltAdaBoostDecisionBound(X_, y_, trees, a,labels):
    pos = y_ == 1
    neg = y_ == -1
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(-0.2, 0.7, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

    Z_ = adaBoostPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], trees, a,labels).reshape(X_tmp.shape)
    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)

    plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
    plt.legend()
    plt.show()


def main():
    path='../DataSet/watermelon3_0a_Ch.txt'
    X,y,labelSet = loadDataSet(path)
    y[y == 0] = -1
    # print(X)
    # print(y)
    trees, a = adaBoostTrain(X, y,labelSet)
    pltAdaBoostDecisionBound(X, y, trees, a, labelSet)


main()