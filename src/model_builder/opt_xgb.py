import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv('../../data/no_price_feature_selected/RandomUnderSampler.csv')
head = 'click_timestamp,nb_clicks_1week,audience_id,product_brand,product_category3,product_category4,product_category5,product_category6,product_country,product_id,partner_id'.split(',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
house_dmatrix = xg.DMatrix(data=X, label=y)

base_params = {
    'learning_rate': np.arange(0.01, 0.3, 0.05),
    'n_estimators': [x for x in range(90, 110, 5)],
    'gamma': [float(x) for x in range(0, 10)],
    'max_depth': [x for x in range(8, 20, 2)],
    'subsample': np.arange(0.5, 0.8, 0.1),
    'min_child_weight': [x for x in range(20)],
    'max_delta_step': [x for x in range(20)]
}

params = {
    'base_score': np.arange(0.2, 0.8, 0.1),
    'learning_rate': np.arange(0.1, 0.4, 0.1),
    'gamma': [float(x) for x in range(5)],
    'max_depth': [x for x in range(10)],
    'min_child_weight': [x for x in range(10)],
    'n_estimators': np.arange(70, 150, 10),
    'subsample': np.arange(0.1, 1.1, 0.1)
}

model = xg.XGBClassifier(use_label_encoder=False, verbosity=0, booster='gbtree')

#search = GridSearchCV(estimator=model, param_grid=params, scoring='recall', cv=2, verbose=100, n_jobs=-1)
search = RandomizedSearchCV(estimator=model, param_distributions=params, scoring=['recall', 'balanced_accuracy'],
                            cv=2, n_iter=40, n_jobs=-1, verbose=1000, refit='recall')

search.fit(X, y)
print(search)
print(search.best_params_)
print(search.best_score_)
print(np.sqrt(np.abs(search.best_score_)))

print('Best params:')
print('recall:\t', search.best_params_)
