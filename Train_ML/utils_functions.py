import copy
import numpy as np


def avg_scores(cv_scores):
    mcc = np.empty(len(cv_scores))
    acc = np.empty(len(cv_scores))
    confusion_norm = np.empty((len(cv_scores),2,2))
    confusion = np.empty((len(cv_scores),2,2))
    
    for i, s in enumerate(cv_scores):
        mcc[i] = s['test_MCC']
        acc[i] = s['test_ACC']
        confusion[i] = s['test_confusion_matrix']
        confusion_norm[i] = (confusion[i] / confusion[i].sum()) * 100
    
    return acc, mcc, confusion, confusion_norm
   

def train_and_score_model(model, Xdata, ydata, scoring):
    model = copy.deepcopy(model)
    Xdata = copy.deepcopy(Xdata)
    ydata = copy.deepcopy(ydata)
    scoring = copy.deepcopy(scoring)
    
    def score(train_idx, test_idx):
        Xtrain, ytrain = Xdata[train_idx], ydata[train_idx]
        Xtest, ytest = Xdata[test_idx], ydata[test_idx]
        
        model.fit(Xtrain, ytrain)
        
        scores = {}
        for key, f in  scoring.items():
            scores["test_" + key] = f(model, Xtest, ytest)
            scores["train_" + key] = f(model, Xtrain, ytrain)
        return scores
    return score


def print_scores(cv_scores):
    """ """
    acc, mcc, conf, conf_norm = avg_scores(cv_scores)
    
    # 
    print(f"RF(acc): {acc.mean():.3f} +/- {acc.std():.3f}")
    print(f"RF(mcc): {mcc.mean():.3f} +/- {mcc.std():.3f}")
    
    print()
    mean_norm = conf_norm.mean(axis=0)
    std_norm = conf_norm.std(axis=0)
    print('confusion_matrix mean')
    print(mean_norm.round(4))
    print('\nconfusion_matrix std')
    print(std_norm.round(4))

    print()
    to_compute = mean_norm[0,1] + mean_norm[1,1]

    missed_of_conv = np.array([x[1,0] / sum(x[1,:]) for x in conf]).mean()

    print(f"To compute: {to_compute:.2f}% of all")
    print(f"missing: {missed_of_conv*100:.2f} % of converged")