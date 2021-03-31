import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.mixture import GaussianMixture
from sklearn import metrics

def main():
    x, y = transformData()
    em(x,y)


def transformData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])
    return x,y


def em(x,y): 
    y_predict = GaussianMixture(n_components=2).fit_predict(x)
    print(metrics.rand_score(y, y_predict))


main()