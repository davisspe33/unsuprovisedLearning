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


    transformer = NMF(n_components=7)
    X_transformed = transformer.fit_transform(x)
    return(X_transformed,y)

# transformData()