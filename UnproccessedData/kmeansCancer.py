import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import validation_curve
import datetime


def main():
    x, y = transformData()
    kmean(x,y)
    plotnCluster(x,y)


def transformData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])
    return x,y

def kmean(x,y): 
    y_predict = KMeans(n_clusters=2).fit_predict(x)
    print(metrics.rand_score(y, y_predict))


def plotnCluster(x,y):
    param_range= np.linspace(1, 100, num=100)
    accuracy=[]
    times=[]
    for n in range(1,11):
        start = datetime.datetime.now()   
        y_predict = KMeans(n_clusters=n).fit_predict(x)
        y_test_accuracy = metrics.rand_score(y, y_predict)
        stop = datetime.datetime.now()
        accuracy.append(y_test_accuracy*100)
        times.append(((stop - start).total_seconds()))
  
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    axes[0].grid()
    param_range=np.linspace(1, 10, len(accuracy))
    axes[0].plot(param_range, accuracy, label="kmeans",color="blue", lw=2)
    axes[0].legend(loc="best")
    axes[0].set_xlabel("number of clusters")
    axes[0].set_ylabel("Accuracy score %")
    axes[1].grid()
    param_range=np.linspace(1, 10, len(times))
    axes[1].plot(param_range, times, label="kmeans",color="blue", lw=2)
    axes[1].legend(loc="best")
    axes[1].set_xlabel("number of clusters")
    axes[1].set_ylabel("Time in Seconds")
    plt.suptitle('Cancer')
    plt.show()

main()
    