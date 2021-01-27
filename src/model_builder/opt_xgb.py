import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv('../../data/criteo/balanced.csv', sep=',')
head = 'user_id,partner_id,nb_clicks_1week,product_price,product_age_group,device_type,audience_id,product_gender,click_timestamp,product_id,product_country,product_brand,product_title,product_category6,product_category5,Sales'.split(',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
house_dmatrix = xg.DMatrix(data=X, label=y)

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
print('f1:\t', search.best_params_)
