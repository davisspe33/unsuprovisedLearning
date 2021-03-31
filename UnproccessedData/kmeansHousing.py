import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

def main():
    x, y = transformData()
    kmean(x,y)


def transformData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']
    y = y.astype('int')
    return x,y

def kmean(x,y): 
    y_predict = KMeans(n_clusters=2).fit_predict(x)
    print(metrics.rand_score(y, y_predict))


main()