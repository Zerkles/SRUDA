import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from tornado.process import task_id

data = pd.read_csv('../../data/no_price_feature_selected/RandomUnderSampler.csv')
head = 'click_timestamp,nb_clicks_1week,audience_id,product_brand,product_category3,product_category4,product_category5,product_category6,product_country,product_id,partner_id'.split(
    ',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
X[np.isnan(X)] = 0

house_dmatrix = xg.DMatrix(data=X, label=y)

base_params = {
    'depth': [4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.02, 0.03, 0.04],
    'n_estimators': np.arange(50, 100, 10),
    'subsample': np.arange(0.5, 0.801, 0.1)
}

param_random_gb = {
    'n_estimators': np.arange(100, 130, 5),
    'max_depth': [7],
    'learning_rate': np.arange(0.01, 0.03, 0.005),
    'reg_lambda': np.arange(3.5, 5.0, 0.5),
    'scale_pos_weight': np.arange(25.0, 30.0, 1.0),
    'border_count': np.arange(115, 130, 5),
}
gb = CatBoostClassifier(loss_function='Logloss', eval_metric='Logloss', leaf_estimation_method='Newton',
                        grow_policy='SymmetricTree', task_type='GPU')
print(gb)

params = {
    'iterations': [1600],
    'l2_leaf_reg': [2],
    'subsample': [0.6],
    'depth': [11],
    'border_count': [200],
    'score_function': ['L2', 'Cosine']
}

search = RandomizedSearchCV(estimator=gb, param_distributions=params, n_iter=5,
                            scoring=['recall', 'balanced_accuracy'], cv=2, verbose=100, n_jobs=-1, refit='recall')
# search = GridSearchCV(estimator=gb, param_grid=params, n_jobs=-1, scoring=['recall', 'balanced_accuracy'], cv=2, verbose=100, refit='recall')

search.fit(X, y)
print()
print(search.best_estimator_)
print(search.best_score_)
print("Best parameter: ", search.best_params_)
