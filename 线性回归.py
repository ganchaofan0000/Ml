import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def splitarr(arr):
    arrtrain=arr[len(arr)//3:]
    arrtest=arr[:len(arr)//3]
    return arrtrain,arrtest

def readcsv(path):
    datalist=[]
    with open(path,'r') as csv_file:
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        for one_line in csv_reader_lines:
            datalist.append(one_line)  # 将读取的csv分行数据按行存入列表list中
    arr=[]
    for a in datalist[1:]:
        a = list(map(float,a))
        arr.append(a)
    return arr

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
    X_norm = (x - mu1) / sigma1

    return X_norm


def computeCostMulti(x, y, theta):
    """
    计算损失函数
    :param X:
    :param y:
    :param theta:
    :return:
    """
    m = x.shape[0]
    costs = x.dot(theta) - y
    total_cost = costs.transpose().dot(costs) / (2 * m)
    return total_cost[0][0]


def gradientDescentMulti(x, y, theta, alpha, iterNum):
    """
    梯度下降实现
    :param X:
    :param y:
    :param theta:
    :param alpha:
    :param iterNum:
    :return:
    """
    m = len(x)

    J_history = list()

    for i in range(0, iterNum):
        costs = x.dot(theta) - y
        theta = theta - np.transpose(costs.transpose().dot(x) * (alpha / m))

        J_history.append(computeCostMulti(x, y, theta))

    return theta, J_history


def learningRatePlot(x_norm, y):
    """
    不同学习速率下的梯度下降比较
    :param x_norm:
    :param y:
    :return:
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure()
    iter_num = 50
    # 如果学习速率取到3，损失函数的结果随着迭代次数增加而发散，值越来越大，不太适合在同一幅图中展示
    for i, al in enumerate([0.01, 0.03, 0.1, 0.3]):
        ta = np.zeros((x_norm.shape[1], 1))
        ta, J_history = gradientDescentMulti(x_norm, y, ta, al, iter_num)
        plt.plot([i for i in range(len(J_history))], J_history, colors[i], label=str(al))

    plt.title("learning rate")
    plt.legend()
    plt.show()


def normalEquation(x, y):
    """
    正规方程实现
    :param X:
    :param y:
    :return:
    """
    return np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)


if __name__ == '__main__':
    # 读取数据
    train_data_path = r'./train_dataset.csv'
    test_data_path=r'./test_dataset.csv'

    # 训练数据整理
    train0,test0=splitarr(readcsv(train_data_path))
    train = np.array(train0)
    x_train = train[:,0:-1]
    y_train = train[:,-1].reshape(x_train.shape[0],1)
    # 训练数据标准化数据标准化
    xtrain_norm= featureNormalization(x_train)
    ones = np.ones((xtrain_norm.shape[0], 1))
    # 假设函数中考虑截距的情况下，给每个样本增加一个为1的特征
    xtrain_norm = np.c_[ones, xtrain_norm]
    # 初始化theta0
    theta0 = np.zeros((xtrain_norm.shape[1], 1))


    #测试数据整理
    test = np.array(test0)
    x_test = test[:, 0:-1]
    y_test = test[:, -1].reshape(x_test.shape[0], 1)
    # 测试数据标准化数据标准化
    xtest_norm = featureNormalization(x_test)
    ones = np.ones((xtest_norm.shape[0], 1))
    # 假设函数中考虑截距的情况下，给每个样本增加一个为1的特征
    xtest_norm = np.c_[ones, xtest_norm]

    # 分析梯度下降的学习率alpha，以及迭代次数iterNum
    cost=100000
    alpha_best=0.0
    iterNum_best=0
    theta_best=np.array([])
    for i in range(1,100):
        for j in range(5,100):
            alpha=i/100
            iterNum=j
            theta, J_history=gradientDescentMulti(xtrain_norm,y_train,theta0,alpha,iterNum)
            cost1=computeCostMulti(xtest_norm,y_test,theta)
            if(cost1<cost):
                cost=cost1
                alpha_best=alpha
                iterNum_best=iterNum
                theta_best=theta
    # print(cost)
    # print(alpha_best)
    # print(iterNum_best)
    # print(theta_best)




    # 解决实际
    test = np.array(readcsv(test_data_path))
    x = test[:, 1:]
    # 数据标准化
    x_norm = featureNormalization(x)
    ones = np.ones((x_norm.shape[0], 1))
    # 假设函数中考虑截距的情况下，给每个样本增加一个为1的特征
    x_norm = np.c_[ones, x_norm]
    # 输出theta
    #print("theta的取值为%s" % (theta))
    price=x_norm.dot(theta_best)
    print("价格为%s" %(price.shape[0]))
    with open("price3.csv", "w",encoding='utf-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        writer.writerow(["ID", "value"])
        for i in range(0,price.shape[0]):
            writer.writerow(["id_%s"%(i+1),price[i,0]])
