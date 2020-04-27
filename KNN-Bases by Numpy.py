#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import operator

#首先导入鸢尾花数据集文件 
def get_excle():
    head = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Species']
    f = open("D:/DeepNeuralNetworks_learning/day02(KNN based by Numpy)/iris.csv")
    iris = pd.read_csv(f, names=head)
    #清除所有重复行,inplace表示在原数据上修改，不需要重新赋值
    iris.drop_duplicates(inplace=True)
    #清除Species前缀
    iris["Species"] = iris.Species.apply(lambda x: x.split("-")[1])
    #将Species映射为数字形式
    #对文本进行处理，将Species列的文本映射成数值类型
    iris['Species'] = iris['Species'].map({'virginica':0,'setosa':1,'versicolor':2})
    return iris
iris = get_excle()


# In[110]:


#将鸢尾花的数据分开为测试集与训练集
def irisClassFication():
    #随机抽选10个数据形成测试集
    X_test = iris.sample(36)
    #总的数据集干掉测试集就是训练集
    X_train = iris.drop(X_test.index)
    return X_train, X_test


# In[111]:


X, Y = irisClassFication()


# In[114]:


#KNN近邻算法分析
def knn(X_train, X_test, y_train=0, y_test=0, K=1):
    #获取训练集的行数
    totalrow = X_train.shape[0]
    #计算测试数据正确匹配该数据原lable
    corret_num = 0
    #循环测试数据集，计算每一个测试数据与训练集的欧氏距离
    for i in range(len(X_test)):
        x_test = np.tile(X_test[i],(totalrow,1)) - X_train
        #两点间欧氏距离,axis=1代表一行数据之和
        o_distance = np.sqrt((x_test ** 2).sum(axis=1))
        #重新由小到大排序，按照原来的index返回
        sort_distance = o_distance.argsort()
        #取到前K个最小的距离的index
        min_distance = sort_distance[:K]
        tmp = {'virginica':0,'setosa':0,'versicolor':0}
        for j in min_distance:
            if y_train[j] == 0:
                tmp["virginica"] += 1
            elif y_train[j] == 1:
                tmp["setosa"] += 1
            else:
                tmp["versicolor"] += 1
        if tmp["virginica"] >= 2:
            voteLable = 0
        elif tmp["setosa"] >= 2:
            voteLable = 1
        elif tmp["versicolor"] >= 2:
            voteLable = 2
        else:
            voteLable = y_train[min_distance[0]]
        print(voteLable,y_test[i],X_test[i])


# In[122]:


knn(X.drop(columns=["Species"]).values, Y.drop(columns=["Species"]).values, X["Species"].values, Y["Species"].values, K=3)


# In[120]:


#数据可视化
import seaborn as sns
import matplotlib.pyplot as plt

#sns初始化
sns.set()

#设置散点图x轴与y轴以及data参数
sns.relplot(x='sepal-length', y='sepal-width',  style="Species", data=Y)
plt.title('SepalLengthCm and SepalWidthCm data analysize')


# In[121]:


sns.relplot(x='sepal-length', y='sepal-width',  style="Species", data=X)
plt.title('SepalLengthCm and SepalWidthCm data analysize')

