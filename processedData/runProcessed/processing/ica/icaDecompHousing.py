import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn import metrics
from sklearn.decomposition import FastICA

def transformData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']
    y = y.astype('int')

    transformer = FastICA(n_components=10)
    X_transformed = transformer.fit_transform(x)
    return(X_transformed,y)

def graphMixing(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    print(x.shape)
    variance=[]
    for i in range(1,14):
        transformer = FastICA(n_components=i)
        transformer.fit_transform(x)
        l = transformer.mixing_
        variance.append(sum(l[0])/len(l[0]))
    
    plt.grid()
    param_range=np.linspace(1, 13, len(variance))
    plt.plot(param_range, variance, label="ICA",color="green", lw=2)
    plt.legend(loc="best")
    plt.xlabel("number of features")
    plt.ylabel("Mixing")
    plt.suptitle('HousingICA')
    plt.show()

#graphMixing()
# transformData()