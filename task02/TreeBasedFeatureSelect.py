from sklearn.ensemble import RandomForestClassifier
from data_loader_extended import create_folds
from sklearn.feature_selection import SelectFromModel


def treeBasedFeatureSelect():
    X_fold, y_fold, X_validate, y_validate, X_test, y_test = create_folds()

    clf = RandomForestClassifier(max_depth=None, min_samples_leaf=5, min_samples_split=3, n_estimators=100)
    clf.fit(X_fold, y_fold)

    sfm = SelectFromModel(clf, threshold='mean')

    sfm.fit(X_fold, y_fold)

    return sfm.get_support(indices=True)

