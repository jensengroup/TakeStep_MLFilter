import numpy as np

from utils_functions import train_and_score_model, print_scores

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    make_scorer,
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier

import lightgbm as lgb


scores = {
    "MCC": make_scorer(matthews_corrcoef),
    "ACC": make_scorer(accuracy_score),
    "confusion_matrix": make_scorer(confusion_matrix),
}


def _load_data():
    with open("processed_data/xtrain.npy", "rb") as ftrain:
        X = np.load(ftrain).astype(np.int8)

    with open("processed_data/ytrain.npy", "rb") as ytrain:
        y = np.load(ytrain).astype(np.int8)

    print("- # converged: ", y.sum())
    print("- # not converged fragments: ", len(np.where(y == 0)[0]))
    print("- ratio converged/total {:.3f}".format(y.sum() / len(y)))

    return X, y


def decison_tree(X, y, folds=5):
    """ """
    tree_clf = DecisionTreeClassifier(random_state=31)
    tree_model = train_and_score_model(tree_clf, X, y, scores)
    skf = StratifiedKFold(n_splits=folds)

    tree_cv_scores = [tree_model(*data_idx) for data_idx in skf.split(X, y)]

    print("DecisionTreeClassifier")
    print_scores(tree_cv_scores)
    print()


def std_RF(X, y, n_jobs=5, folds=5):
    """ """
    clf_rf = RandomForestClassifier(random_state=31, n_jobs=n_jobs)
    cv_rf_model = train_and_score_model(clf_rf, X, y, scores)

    skf = StratifiedKFold(n_splits=folds)
    rf_cv_scores = [cv_rf_model(*data_idx) for data_idx in skf.split(X, y)]

    print("Random Forest Classifier")
    print_scores(rf_cv_scores)
    print()


def RF_w_weights(X, y, n_jobs=5, folds=5):
    """ """
    clf_rf_cw = RandomForestClassifier(
        random_state=31, class_weight={0: 1, 1: 1.5}, n_jobs=n_jobs
    )
    cv_rf_cw_model = train_and_score_model(clf_rf_cw, X, y, scores)

    skf = StratifiedKFold(n_splits=folds)
    rf_cw_cv_scores = [cv_rf_cw_model(*data_idx) for data_idx in skf.split(X, y)]

    print("Random Forest Classifier with weights")
    print_scores(rf_cw_cv_scores)
    print()


def Balanced_RF(X, y, n_jobs=5, folds=5):
    clf_balanced_rf = BalancedRandomForestClassifier(random_state=31, n_jobs=n_jobs)
    cv_balanced_rf_model = train_and_score_model(clf_balanced_rf, X, y, scores)

    skf = StratifiedKFold(n_splits=folds)
    balanced_rf_cv_scores = [
        cv_balanced_rf_model(*data_idx) for data_idx in skf.split(X, y)
    ]

    print("Balanced Random ForestClassifier")
    print_scores(balanced_rf_cv_scores)
    print()


def lightgbm_w_oversampling(X, y, n_jobs=5, folds=5):
    ros = RandomOverSampler(random_state=31)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    lightgbm_model = lgb.LGBMClassifier(
        objective="binary", random_state=31, n_jobs=n_jobs
    )
    cv_lightgbm_model = train_and_score_model(
        lightgbm_model, X_resampled, y_resampled, scores
    )

    skf = StratifiedKFold(n_splits=folds)
    lightgbm_model_cv_scores = [
        cv_lightgbm_model(*data_idx) for data_idx in skf.split(X, y)
    ]

    print("LightGBM with oversampling")
    print_scores(lightgbm_model_cv_scores)
    print()


if __name__ == "__main__":

    n_cpus = 4

    xtrain, ytrain = _load_data()

    decison_tree(xtrain, ytrain, folds=5)
    std_RF(xtrain, ytrain, n_jobs=n_cpus, folds=5)
    RF_w_weights(xtrain, ytrain, n_jobs=n_cpus, folds=5)
    Balanced_RF(xtrain, ytrain, n_jobs=n_cpus, folds=5)
    lightgbm_w_oversampling(xtrain, ytrain, n_jobs=5, folds=5)
