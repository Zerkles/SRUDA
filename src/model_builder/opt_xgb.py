import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.balancing.data_controller import DataController

data = DataController.read_categorized_criteo('../../data/balanced_csv/RandomUnderSampler.csv')
X, y = DataController.split_data_on_x_y(data)

X = X.values
y = y.values.ravel()

base_params = {
    'learning_rate': np.arange(0.001, 0.2, 0.001),
    'n_estimators': [x for x in range(100, 300, 10)],
    'subsample': np.arange(0.1, 0.99, 0.05),
    'gamma': [x for x in range(1, 20)],
    'max_depth': [x for x in range(20)],
    'min_child_weight': [x for x in range(20)],
    'max_delta_step': [x for x in range(20)]
}

params = {
    'learning_rate': [0.003],
    'n_estimators': [220],
    'subsample': [0.8],
    'gamma': [9, 10, 11],
    'max_depth': [6],
    'min_child_weight': [7, 8, 9],
    'max_delta_step': [1]
}

model = xg.XGBClassifier(use_label_encoder=False, verbosity=0, booster='gbtree')

search = GridSearchCV(estimator=model, param_grid=params, scoring='recall', cv=2, verbose=100, n_jobs=-1)

search.fit(X, y)
print(search)
print(search.best_params_)
print(search.best_score_)
print(np.sqrt(np.abs(search.best_score_)))

print('Best params:')
print(search.best_params_)
