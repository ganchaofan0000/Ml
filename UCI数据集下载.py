import pandas as pd

def ReadAndSaveDataByPandas(target_url=None, save=False):
    wine = pd.read_csv(target_url)
    if save == True:
        wine.to_csv("DataSet/glass.csv", index=False)


target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"  # 一个玻璃的多分类的数据集
ReadAndSaveDataByPandas(target_url, True)