import numpy as np
import math

# 读取数据氙气病症与马死亡的数据
def readtxt(path):
    """
    读取数据氙气病症与马死亡的数据
    :param path:
    :return:
    """
    datalist=[]
    with open(path, 'r') as f:
        for line in f:
            datalist.append(list(line.strip('\n').split('\t')))
    arr=[]
    for a in datalist:
        a = list(map(float,a))
        arr.append(a)
    return arr


# 处理数据
def assessmentmethod(traindata) :
    train = traindata[len(traindata) // 3:]
    test = traindata[:len(traindata) // 3]
    # 训练集
    train = np.mat(np.array(train))
    x_train = train[:, 0:-1]
    y_train = train[:, -1].reshape(x_train.shape[0], 1)
    # 训练数据标准化数据标准化
    xtrain_norm = featureNormalization(x_train)
    ones = np.ones((xtrain_norm.shape[0], 1))
    # 假设函数中考虑截距的情况下，给每个样本增加一个为1的特征
    xtrain_norm = np.c_[ones, xtrain_norm]

    # 验证集
    test = np.mat(np.array(test))
    x_test = test[:, 0:-1]
    y_test = test[:, -1].reshape(x_test.shape[0], 1)
    # 验证数据标准化数据标准化
    xtest_norm = featureNormalization(x_test)
    ones = np.ones((xtest_norm.shape[0], 1))
    # 假设函数中考虑截距的情况下，给每个样本增加一个为1的特征
    xtest_norm = np.c_[ones, xtest_norm]

    return xtrain_norm,y_train,xtest_norm,y_test


def computecorrectrate(x, y, theta):
    # 计算正确率
    m = x.shape[0]
    h=sigmoid(x.dot(theta))
    costs = h - y
    correct=0
    for a in costs:
        if abs(a[0])<0.5:
            correct+=1
    rate=correct/costs.shape[0]
    return rate


def featureNormalization(x):
    """
    数据标准化
    :param X:
    :return:
    """
    # 平均值
    mu = np.mean(x,axis=0)
    # ddof的设置为1表示无偏的标准差
    sigma = np.std(x, axis=0, ddof=1)
    # 特征缩放
    mu1=np.tile(mu,(x.shape[0],1))
    sigma1=np.tile(sigma,(x.shape[0],1))
    x_norm = (x - mu1) / sigma1

    return x_norm


# S函数
def sigmoid(X):
    return 1/(1+np.exp(-X))


# 梯度上升函数
def gradientrise(x,y,alpha, iterNum):

    # 初始化theta0
    theta = np.zeros((x.shape[1], 1))
    for i in range(0, iterNum):
        costs = y - sigmoid(x.dot(theta))
        theta = theta + alpha * x.transpose().dot(costs)
    return theta


# 确定最好的alpha, iterNum
def optimization(x_train,y_train,x_test,y_test):
    # 分析梯度下降的学习率alpha，以及迭代次数iterNum
    correct = 0.0
    alpha_best = 0.0
    iterNum_best = 0
    theta_best = np.array([])
    for i in range(1, 100):
        for j in range(1, 100):
            alpha = i / 10000
            iterNum = j
            theta = gradientrise(x_train, y_train, alpha, iterNum)
            correct1 = computecorrectrate(x_test, y_test, theta)
            if (correct1 > correct):
                correct = correct1
                alpha_best = alpha
                iterNum_best = iterNum
                theta_best = theta
    print("测试正确率为%s"%correct)
    return theta_best
def main():
    train_path='../DataSet/horseColicTraining.txt'
    test_path='../DataSet/horseColicTest.txt'
    train_data=readtxt(train_path)

    x_train, y_train, x_test, y_test=assessmentmethod(train_data)
    theta_best=optimization(x_train,y_train,x_test,y_test)

    # 测试实例
    test_data=readtxt(test_path)
    # 训练集
    test = np.mat(np.array(test_data))
    x = test[:, 0:-1]
    y = test[:, -1].reshape(x.shape[0], 1)
    # 训练数据标准化数据标准化
    x_norm = featureNormalization(x)
    ones = np.ones((x_norm.shape[0], 1))
    # 假设函数中考虑截距的情况下，给每个样本增加一个为1的特征
    x_norm = np.c_[ones, x_norm]
    correct = computecorrectrate(x_norm, y, theta_best)
    print("实际正确率%s"%correct)


main()