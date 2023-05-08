import numpy as np
from sklearn.model_selection import StratifiedKFold

def load():
    X = np.concatenate((np.load('../dataset/X0.npy'), 
        np.load('../dataset/X1.npy'), 
        np.load('../dataset/X2.npy'), 
        np.load('../dataset/X3.npy'))) #feature matrix
    y = np.load('../dataset/y.npy') #labels
    a = np.load('../dataset/a.npy') #agreement as a probability

    #group agreement into bins
    a_digit = np.digitize(a, [0.5, 0.6, 0.7, 0.8, 0.9])
    #generate unique class-agreement pair
    stratify_criterion = np.array([label * 10 + agreement for label, agreement in zip(y, a_digit)])
    stratify_criterion = np.unique(stratify_criterion, return_inverse = True)[1]   
    
    skf = StratifiedKFold(int(1/0.2)) #20% test set
    train_index, test_index = next(skf.split(X, stratify_criterion))

    return X[train_index], y[train_index], stratify_criterion[train_index], X[test_index], y[test_index]
    