import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams['axes.unicode_minus'] = False

def set_ax_gray(ax):
    # 背景颜色
    ax.patch.set_facecolor("gray")
    # 背景颜色透明度
    ax.patch.set_alpha(0.1)
    # 设置边界线看不见
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    # 设置网格线
    ax.grid(axis='y', linestyle='-.')


path = r'../DataSet/watermelon3_0a_Ch.txt'
data = pd.read_table(path, delimiter=' ', dtype=float)

X = data.iloc[:, [0]].values
y = data.iloc[:, 1].values
# 高斯核中的带宽
gamma = 10
# 代价系数
C = 1

ax = plt.subplot()
set_ax_gray(ax)
# 设置散点图
ax.scatter(X, y, color='c', label='data')

for gamma in [1, 10, 100, 1000]:
    svr = svm.SVR(kernel='rbf', gamma=gamma, C=C)
    svr.fit(X, y)
    # reshape(-1, c)-1的作用就在此，自动计算d：d=数组或者矩阵里面所有的元素个数/c, d必须是整数
    ax.plot(np.linspace(0.2, 0.8), svr.predict(np.linspace(0.2, 0.8).reshape(-1, 1)),
            label='gamma={}, C={}'.format(gamma, C))
# 显示label
ax.legend(loc='upper left')
ax.set_xlabel('密度')
ax.set_ylabel('含糖率')

plt.show()