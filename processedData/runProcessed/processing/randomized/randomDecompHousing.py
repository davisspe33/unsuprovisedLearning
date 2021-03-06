import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn import metrics
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def transformData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']
    y = y.astype('int')

    transformer = random_projection.GaussianRandomProjection(n_components=10)
    X_transformed = transformer.fit_transform(x)
    return(X_transformed,y)


# transformData()