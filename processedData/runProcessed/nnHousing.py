import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

import processing.ica.icaDecompHousing as ica
import processing.nmf.nmfDecompHousing as nmf
import processing.pca.pcaDecompHousing as pca
import processing.randomized.randomDecompHousing as randDe
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    nnProcessed()


def nnProcessed(): 
    l = [ica,nmf,pca,randDe]
    module = ["ica","nmf","pca","randDe"]
    count=0
    for z in l:
        print('this is the ' +module[count])
        x, y = z.transformData()
        clf = MLPClassifier(max_iter=400, hidden_layer_sizes=50)
        scores = cross_val_score(clf, x, y, cv=5, verbose=False)
        print(module[count]+ ' neural net cv score')
        print(scores)
        print(module[count] +'  done')
        count+=1
main()