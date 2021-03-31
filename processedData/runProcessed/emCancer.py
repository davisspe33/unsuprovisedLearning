import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics

import processing.ica.icaDecompCancer as ica
import processing.nmf.nmfDecompCancer as nmf
import processing.pca.pcaDecompCancer as pca
import processing.randomized.randomDecompCancer as randDe

def main():
    emProcessed()


def emProcessed(): 
    l = [ica,nmf,pca,randDe]
    module = ["ica","nmf","pca","randDe"]
    count=0
    for z in l:
        print('this is the ' +module[count])
        x, y = z.transformData()
        y_predict = GaussianMixture(n_components=2).fit_predict(x)
        score = metrics.rand_score(y, y_predict)
        print(module[count] + ' em rand score')
        print(score)
        print(module[count] +'  done')
        count+=1
main()