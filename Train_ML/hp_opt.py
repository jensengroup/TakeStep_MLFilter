import optuna

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import  matthews_corrcoef, make_scorer
from sklearn.model_selection import cross_validate

from models import _load_data


X, y = _load_data()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
    }

    clf = BalancedRandomForestClassifier(random_state=31, **params, n_jobs=4)
    scores = cross_validate(clf, X, y, n_jobs=1, scoring=make_scorer(matthews_corrcoef), cv=5)
    return scores['test_score'].mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
