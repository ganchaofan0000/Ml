# import pandas as pd
# import csv
# path = 'DataSet/西瓜数据集2.txt'
# reader0=pd.read_csv(path,encoding='gbk')
#
# # 删除名称为‘编号’的的列
# reader0.drop('编号',axis=1,inplace=True)
# # 获得列名,列表形式
# label=list(reader0.columns.values)
# # 以list形式获得数据
# data=reader0.values.tolist()
# # 以字典列表的形式获得数据,其中orient可以选择六种的转换类型，
# # 分别对应于参数 ‘dict’, ‘list’, ‘series’, ‘split’, ‘records’, ‘index’
# da = reader0.to_dict(orient='records')
# print(data)
# with open(path,'r',encoding='gbk') as f:
#     reader=csv.reader(f)
#     labels=next(reader)
#     print(labels)

from collections import Counter
import numpy as np
words = [
    'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
    'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
    'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
    'my', 'eyes', "you're", 'under'
]
words=np.array(words)

# print(words.shape)
# print(Counter(words).most_common())


# a = np.array([3, 2, 1])
# b=np.array([2,-2,2])
# # itemindex = np.argwhere(a == 3)
# # print(a)
# # index=itemindex[0][0]
# # print(index)
# # print(a[index])
# print(a.dot(a))

import random
# for i in range(10):
#     print(random.randint(0, 9))

# a=[]
# d=np.array([[1,2,3],[4,5,6]])
#
# a.append(d[0,:])
# a.append(d[1,:])
# print(np.array(a))


import numpy as np
from collections import Counter
import math
# rows = 10
# cols = 9
# arr = np.random.random_integers(1, 10, (10, 9))  # 生成整数数组
# print("二维数组元素：", arr)
#
# result = [Counter(arr[:, i]).most_common(1)[0] for i in range(cols)]
# print("按列统计的结果为：", result)  # 显示的结果中第一个是数字，第二个是这个数字出现的次数
# result = [Counter(arr[i, :]).most_common(1)[0] for i in range(rows)]
# print("按行统计的结果为：", result)
# print(math.log2(3))
# print(int(math.log2(3)))
#
# list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# slice = random.sample(list, 5)  #从list中随机获取5个元素，作为一个片断返回
# print (slice)
# print (list)


li=[1,2,3,4,5,6]
random.shuffle(li)
print(li)
