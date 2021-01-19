from builtins import set

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import time

data = pd.read_csv('../../data/balanced_csv/RUS.csv', sep=',')
head = 'click_timestamp,nb_clicks_1week,product_price,audience_id,product_brand,product_category3,product_category4,product_category5,product_category6,product_country,product_id,partner_id,Sales'.split(',')
print(head)


X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
X[np.isnan(X)] = 0

depths = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]

min_sample_splits = [x/10 for x in range(1, 9)] + [x for x in range(1, 10)]

max_features = ['auto', 'sqrt', 'log2', None]

params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': depths,
    'min_samples_split': min_sample_splits,
    'min_weight_fraction_leaf': [x/10 for x in range(1, 5)],
    'max_features': max_features,
    'random_state': np.arange(0, 50, 1)
}

model = DecisionTreeClassifier()
print(model)


#search = GridSearchCV(estimator=model, param_grid=params, scoring='recall', cv=2, verbose=100, n_jobs=-1)
search = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='recall', n_iter=40, cv=2, n_jobs=-1,
                            verbose=100)
search.fit(X, y)

print('============================')
print(search)
print(search.best_params_)
print(search.best_score_)
print(np.sqrt(np.abs(search.best_score_)))
print('Best params:')
print('f1:\t', search.best_params_)

