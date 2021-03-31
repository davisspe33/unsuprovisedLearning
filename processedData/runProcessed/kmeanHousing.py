import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn import metrics

import processing.ica.icaDecompHousing as ica
import processing.nmf.nmfDecompHousing as nmf
import processing.pca.pcaDecompHousing as pca
import processing.randomized.randomDecompHousing as randDe

def main():
    kmeanProcessed()


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
main()