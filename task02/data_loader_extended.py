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


def create_folds(top_features: int = None):
    X_train, y_train, stratify_criterion, X_test, y_test = load()

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    correlation_coeffs = np.corrcoef(X.T, y)
    feature_correlations = np.abs(correlation_coeffs[:-1, -1])
    best_feature_indices = np.argsort(feature_correlations)[::-1]

    # Top 20 features
    if top_features:
        top_feature_indices = best_feature_indices[:top_features]
    else:
        top_feature_indices = best_feature_indices
    X_train = X_train[:, top_feature_indices]
    X_test = X_test[:, top_feature_indices]


    X_fold, y_fold, X_validate, y_validate = None, None, None, None
    skf = StratifiedKFold(5)
    for i, (train_index, validate_index) in enumerate(skf.split(X_train, stratify_criterion)):
        X_fold = X_train[train_index]
        y_fold = y_train[train_index]
        X_validate = X_train[validate_index]
        y_validate = y_train[validate_index]



    print(X_fold.shape, y_train.shape, stratify_criterion.shape, X_test.shape, y_test.shape)
    return X_fold, y_fold, X_validate, y_validate, X_test, y_test
