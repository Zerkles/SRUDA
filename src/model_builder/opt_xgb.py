import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv('../../data/balanced_csv/RUS.csv', sep=',')
head = 'click_timestamp,nb_clicks_1week,product_price,audience_id,product_brand,product_category3,product_category4,product_category5,product_category6,product_country,product_id,partner_id,Sales'.split(',')
print(head)


X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
X[np.isnan(X)] = 0

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

#search = GridSearchCV(estimator=model, param_grid=params, scoring='recall', cv=2, verbose=100, n_jobs=-1)
search = RandomizedSearchCV(estimator=model, param_distributions=base_params, scoring='recall', cv=2, n_iter=40,
                            n_jobs=-1, verbose=100)

search.fit(X, y)
print(search)
print(search.best_params_)
print(search.best_score_)
print(np.sqrt(np.abs(search.best_score_)))

print('Best params:')
print('f1:\t', search.best_params_)
