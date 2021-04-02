import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn import metrics
from sklearn.decomposition import NMF

def transformData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])


    transformer = NMF(n_components=10)
    X_transformed = transformer.fit_transform(x)
    return(X_transformed,y)

def graphError(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    
    variance=[]
    for i in range(1,32):
        transformer = NMF(n_components=i)
        transformer.fit_transform(x)
        l = transformer.reconstruction_err_
        variance.append(l)
    
    plt.grid()
    param_range=np.linspace(1, 32, len(variance))
    plt.plot(param_range, variance, label="NMF",color="green", lw=2)
    plt.legend(loc="best")
    plt.xlabel("number of features")
    plt.ylabel("Reconstruction Error")
    plt.suptitle('Canser NMF')
    plt.show()

#graphError()
#transformData()