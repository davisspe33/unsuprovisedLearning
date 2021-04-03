import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn import metrics
from sklearn.decomposition import PCA

def transformData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']
    y = y.astype('int')

    transformer = PCA(n_components=10)
    X_transformed = transformer.fit_transform(x)
    return(X_transformed,y)

def graphVaraiance(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    print(x.shape)
    variance=[]
    for i in range(1,14):
        transformer = PCA(n_components=i)
        transformer.fit_transform(x)
        l = transformer.explained_variance_ratio_
        variance.append(sum(l)/len(l))
    

    
    plt.grid()
    param_range=np.linspace(1, 13, len(variance))
    plt.plot(param_range, variance, label="PCA",color="green", lw=2)
    plt.legend(loc="best")
    plt.xlabel("number of features")
    plt.ylabel("Varance")
    plt.suptitle('HousingPCA')
    plt.show()

#graphVaraiance()
# transformData()