import numpy as np
a1 = np.array([1,2,3])
a1 = np.mat(a1)
# print(type(a1))
b=np.arange( 10, 30, 5 )
x = np.linspace(0, 2 * np.pi, 100) # useful to evaluate function at lots of points
# print(type(x))
f = np.sin(x)

# 产生一个2*2的对角矩阵

data6=np.mat(np.eye(2,2,dtype=int))

# 生成一个对角线为1、2、3的对角矩阵

a1=[1,2,3]
a2=np.mat(np.diag(a1))

# 求矩阵matrix([[0.5,0],[0,0.5]])的逆矩阵
a1=np.mat(np.eye(2,2)*0.5)
a2=a1.I

# 矩阵转置

a1=np.mat([[1,1],[0,0]])
a2=a1.T

a2=a1.sum(axis=0) #列和，这里得到的是1*2的矩阵
a3=a1.sum(axis=1) #行和，这里得到的是3*1的矩阵
a4=sum(a1[1,:]) #计算第一行所有列的和，这里得到的是一个数值

a1.max()#计算a1矩阵中所有元素的最大值,这里得到的结果是一个数值
a2=max(a1[:,1])#计算第二列的最大值，这里得到的是一个1*1的矩阵
a1[1,:].max()#计算第二行的最大值，这里得到的是一个一个数值

np.max(a1,0)#计算所有列的最大值，这里使用的是numpy中的max函数
np.max(a1,1)#计算所有行的最大值，这里得到是一个矩阵

np.argmax(a1,0)#计算所有列的最大值对应在该列中的索引
np.argmax(a1[1,:])#计算第二行中最大值对应在改行的索引

a=np.mat(np.ones((2,2)))
b=np.mat(np.eye(2))
c=np.vstack((a,b))#按列合并，即增加行数
d=np.hstack((a,b))#按行合并，即行数不变，扩展列数

a=np.mat(np.arange(4).reshape((2,2)))
print(a[1,1])


