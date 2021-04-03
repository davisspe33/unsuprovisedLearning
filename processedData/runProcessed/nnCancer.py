import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
import processing.ica.icaDecompCancer as ica
import processing.nmf.nmfDecompCancer as nmf
import processing.pca.pcaDecompCancer as pca
import processing.randomized.randomDecompCancer as randDe
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    #nnProcessed()
    plotmodelLearn()


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

def plotmodelLearn():
    l = [ica,nmf,pca,randDe]
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    module = ["ica","nmf","pca","randDe"]
    count=0
    for z in l:
        print('this is the ' + module[count])
        x, y = z.transformData()

        train_sizes=np.linspace(.1, 1.0, 5)
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(MLPClassifier(max_iter=400, hidden_layer_sizes=50), x, y, train_sizes=train_sizes, return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        axes[0].plot(train_sizes, train_scores_mean, label="Training score: "+module[count])
        axes[1].plot(train_sizes, test_scores_mean, label="Cross-validation score: "+module[count])
        axes[2].plot(train_sizes, fit_times_mean, label=module[count])
        count+=1

    
    # Plot learning curve
    axes[0].grid()
    axes[0].legend(loc="best")
    axes[0].set_title("Learning Curve")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    # Plot learning curve
    axes[1].grid()
    axes[1].legend(loc="best")
    axes[1].set_title("Learning Curve")
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Score")
    # Plot n_samples vs fit_times
    axes[2].grid()
    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("fit_times")
    axes[2].set_title("Scalability of the model")
    plt.suptitle('NN Cancer')
    plt.show()
main()