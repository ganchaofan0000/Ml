import pandas as pd
import csv
path = 'DataSet/西瓜数据集2.txt'
reader0=pd.read_csv(path,encoding='gbk')

# 删除名称为‘编号’的的列
reader0.drop('编号',axis=1,inplace=True)
# 获得列名,列表形式
label=list(reader0.columns.values)
# 以list形式获得数据
data=reader0.values.tolist()
# 以字典列表的形式获得数据,其中orient可以选择六种的转换类型，
# 分别对应于参数 ‘dict’, ‘list’, ‘series’, ‘split’, ‘records’, ‘index’
da = reader0.to_dict(orient='records')
print(data)
with open(path,'r',encoding='gbk') as f:
    reader=csv.reader(f)
    labels=next(reader)
    print(labels)