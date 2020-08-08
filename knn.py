
import random
import csv
import numpy as np
import pandas as pd

#读取
files=open('data.csv','r')
reader=csv.DictReader(files)
#datas=pd.read_csv('data.csv')
datas=[row for row in reader]
#分组
random.shuffle(datas)

n=len(datas)//3

test_set=datas[0:n]
train_set=datas[n:]

#knn
#距离
def distance(d1,d2):
    res=0

    for key in ("radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean",
    "concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
    "area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"):
        res+=(float(d1[key])-float(d2[key]))**2

    return res**0.5
    
k=6
def knn(data):
    #距离
    res=[
        {"result":train['diagnosis'],"distance":distance(data,train)}
        for train in train_set
    ]
    #排序
    res=sorted(res,key=lambda item:item['distance'])

    #取得K个
    res2=res[0:k]

    #加权平均
    result={'B':0,'M':0}

    #总距离
    sum=0
    for r in res2:
        sum+=r['distance']
    
    for r in res2:
        result[r['result']]+=1-r['distance']/sum
    
    #print(result)
    #print(data['diagnosis'])
    if result['B']>result['M']:
        return 'B'
    else:
        return 'M'

#测试阶段
correct=0
for test in test_set:
    result=test['diagnosis']
    result2=knn(test)

    if result==result2:
        correct+=1
print(correct)
print(len(test_set))

print("准确率：{:.2f}%".format(100*correct/len(test_set)))