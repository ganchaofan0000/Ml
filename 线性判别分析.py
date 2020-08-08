import numpy as np

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

# 获得均值向量u0，u1
def getaverage(data):
    ones=[]
    zeros=[]
    # 将两个类别分割
    for a in data:
        if a[-1]==0.0:
            zeros.append(a)
        else:
            ones.append(a)
    zeros=np.mat(np.array(zeros))
    ones=np.mat(np.array(ones))
    x0 = zeros[:, 0:-1]
    x1 = ones[:, 0:-1]
    # 获取对应的均值向量
    u0 = (np.mean(x0,axis=0)).T
    u1 = (np.mean(x1, axis=0)).T
    return u0,u1,x0,x1



def withmatrix(u0,u1,x0,x1):
    X0 = np.mat(np.zeros((u0.shape[0], u0.shape[0])))
    X1= np.mat(np.zeros((u1.shape[0], u1.shape[0])))
    for i in range(0, x0.shape[0]):
        X0 += (x0[i].T-u0)*((x0[i].T-u0).T)
    for i in range(0, x1.shape[0]):
        X1 += (x1[i].T-u1) * ((x1[i].T - u1).T)
    Sw = X0 + X1
    return Sw


def LDA(u0,u1,Sw):
    w=Sw.I*(u0-u1)
    return w


def main():
    train_path = './horseColicTraining.txt'
    test_path = './horseColicTest.txt'
    train=readtxt(train_path)
    u0,u1,x0,x1 = getaverage(train)
    Sw=withmatrix(u0,u1,x0,x1)
    w=LDA(u0,u1,Sw)
    print (w)
main()