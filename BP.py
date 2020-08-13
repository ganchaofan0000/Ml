import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams['axes.unicode_minus'] = False

def readcsv(path):
    attributeMap={}
    attributeMap['浅白'] = 0
    attributeMap['青绿'] = 0.5
    attributeMap['乌黑'] = 1
    attributeMap['蜷缩'] = 0
    attributeMap['稍蜷'] = 0.5
    attributeMap['硬挺'] = 1
    attributeMap['沉闷'] = 0
    attributeMap['浊响'] = 0.5
    attributeMap['清脆'] = 1
    attributeMap['模糊'] = 0
    attributeMap['稍糊'] = 0.5
    attributeMap['清晰'] = 1
    attributeMap['凹陷'] = 0
    attributeMap['稍凹'] = 0.5
    attributeMap['平坦'] = 1
    attributeMap['硬滑'] = 0
    attributeMap['软粘'] = 1
    attributeMap['否'] = 0
    attributeMap['是'] = 1
    reader=pd.read_csv(path)
    # 删除名称为‘编号’的的列
    reader.drop('编号', axis=1, inplace=True)
    labelset = list(reader.columns.values)
    reader['密度'] = reader['密度'].astype('float')
    reader['含糖率'] = reader['含糖率'].astype('float')
    dataset = reader.values.tolist()
    # dataset = np.array(dataset)
    m,n = np.shape(dataset)
    for i in range(m):
        for j in range(n):
            if dataset[i][j] in attributeMap.keys():
                dataset[i][j]=attributeMap[dataset[i][j]]
            dataset[i][j]=round(dataset[i][j],3)
    return dataset,labelset


def sigmoid(X):
    return 1/(1+np.exp(-X))


def Backpro(X0,Y0,rate,iterNum):

    d = X0.shape[1]  # the dimension of the input vector
    l = 1  # the dimension of the  output vector
    q = d + 1  # the number of the hide nodes
    theta = [random.random() for i in range(l)]  # the threshold of the output nodes
    theta = np.array(theta)
    gamma = [random.random() for i in range(q)]  # the threshold of the hide nodes
    gamma = np.array(gamma)
    # v size= d*q .the connection weight between input and hide nodes
    v = [[random.random() for i in range(q)] for j in range(d)]
    v = np.array(v)
    # w size= q*l .the connection weight between hide and output nodes
    w = [[random.random() for i in range(l)] for j in range(q)]
    w = np.array(w)
    cost=[]
    while (iterNum > 0):
        iterNum -= 1
        sumE = 0
        for i in range(X0.shape[0]):
            alpha = np.dot(X0[i], v)  # p101 line 2 from bottom, shape=1*q
            b = sigmoid(alpha - gamma)  # b=f(alpha-gamma), shape=1*q
            beta = np.dot(b, w)  # shape=(1*q)*(q*l)=1*l
            predictY = sigmoid(beta - theta)  # shape=1*l ,p102--5.3
            E = sum((predictY - Y0[i]) * (predictY - Y0[i])) / 2  # 5.4
            sumE += E  # 5.16
            # p104
            g = predictY * (1 - predictY) * (Y0[i] - predictY)  # shape=1*l p103--5.10
            e = b * (1 - b) * (np.dot(g, w.T))  # shape=1*q , p104--5.15
            w += rate * np.dot(b.reshape(q, 1), g.reshape(1,l))  # 5.11
            theta -= rate * g  # 5.12
            v += rate * np.dot(X0[i].reshape(d, 1), e.reshape(1,q))  # 5.13
            gamma -= rate * e  # 5.14
        cost.append(sumE)
    plt.plot([i for i in range(len(cost))], cost, 'b')
    plt.title("cost")
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.show()
    return theta,gamma,v,w


def predict(iX,theta,gamma,v,w):
    alpha = np.dot(iX, v)  # p101 line 2 from bottom, shape=m*q
    b=sigmoid(alpha-gamma)#b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY=sigmoid(beta - theta)  # shape=m*l ,p102--5.3
    return predictY

def main():
    path='DataSet/watermelon_3.csv'
    dataset,labelset=readcsv(path)
    # 学习率
    dataset = np.array(dataset)
    rate = 0.2
    iterNum = 5000
    X0 = dataset[:, 0:-1]
    Y0 = dataset[:,-1]
    theta, gamma, v, w = Backpro(X0,Y0,rate,iterNum)
    print(predict(X0,theta,gamma,v,w))

main()


