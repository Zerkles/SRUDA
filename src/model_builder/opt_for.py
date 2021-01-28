import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time

start = time.time()
data = pd.read_csv('../../data/no_price_feature_selected/RandomUnderSampler.csv')
head = 'click_timestamp,nb_clicks_1week,audience_id,product_brand,product_category3,product_category4,product_category5,product_category6,product_country,product_id,partner_id'.split(
    ',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X[np.isnan(X)] = 0

default_params = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None,
                  'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
                  'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                  'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False,
                  'random_state': None, 'verbose': 0, 'warm_start': False}

params = {
    'bootstrap': [True],
    'criterion': ['gini'],
    'max_depth': [x for x in range(5, 15)],
    'max_features': ['auto'],
    'n_estimators': [x for x in range(50, 150, 10)],
    'min_samples_split': [x for x in range(5)],
    'min_samples_leaf': [x for x in range(5)]
}

model = RandomForestClassifier()
print(model)

search = RandomizedSearchCV(estimator=model, param_distributions=params, scoring=['recall', 'balanced_accuracy'],
                            n_iter=1000, cv=3,
                            verbose=100, n_jobs=-1, refit='recall')
search.fit(X, y)

print('============================')
print(search)
print(search.best_params_)
print(search.best_score_)
print(np.sqrt(np.abs(search.best_score_)))
print('Best params:')
print('recall:\t', search.best_params_)

print(time.time() - start)
