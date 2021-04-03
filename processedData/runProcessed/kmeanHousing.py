import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn import metrics
import datetime
import processing.ica.icaDecompHousing as ica
import processing.nmf.nmfDecompHousing as nmf
import processing.pca.pcaDecompHousing as pca
import processing.randomized.randomDecompHousing as randDe

def main():
    #kmeanProcessed()
    plotnCluster()


def kmeanProcessed(): 
    l = [ica,nmf,pca,randDe]
    module = ["ica","nmf","pca","randDe"]
    count=0
    for z in l:
        print('this is the ' +module[count])
        x, y = z.transformData()
        y_predict = KMeans(n_clusters=2).fit_predict(x)
        score = metrics.rand_score(y, y_predict)
        print(module[count]+ ' kmean rand score')
        print(score)
        print(module[count] +'  done')
        count+=1

def plotnCluster():
    l = [ica,nmf,pca,randDe]
    module = ["ica","nmf","pca","randDe"]
    count=0
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    
    for z in l:
        accuracy=[]
        times=[]
        print('this is the ' + module[count])
        x, y = z.transformData()
        for n in range(1,50):
            start = datetime.datetime.now()   
            y_predict = KMeans(n_clusters=n).fit_predict(x)
            y_test_accuracy = metrics.rand_score(y, y_predict)
            stop = datetime.datetime.now()
            accuracy.append(y_test_accuracy*100)
            times.append(((stop - start).total_seconds()))
        param_range_a=np.linspace(1, 50, len(accuracy))
        param_range_t=np.linspace(1, 50, len(times))
        axes[0].plot(param_range_a, accuracy, label=module[count], lw=2)
        axes[1].plot(param_range_t, times, label=module[count], lw=2)
        count+=1

    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    axes[0].grid()
    axes[0].legend(loc="best")
    axes[0].set_xlabel("number of clusters")
    axes[0].set_ylabel("Accuracy score %")
    axes[1].grid()
    axes[1].legend(loc="best")
    axes[1].set_xlabel("number of clusters")
    axes[1].set_ylabel("Time in Seconds")
    plt.suptitle('K-means Housing')
    plt.show()
main()