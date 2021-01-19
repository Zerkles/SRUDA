import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier


data = pd.read_csv('../../data/balanced_csv/RUS.csv', sep=',')
head = 'click_timestamp,nb_clicks_1week,product_price,audience_id,product_brand,product_category3,product_category4,product_category5,product_category6,product_country,product_id,partner_id,Sales'.split(',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
X[np.isnan(X)] = 0

param_random_gb = {
    'n_estimators': np.arange(10, 200, 5),
    'max_depth': [x for x in range(10)],
    'learning_rate': np.arange(0.01, 0.1, 0.005),
    'reg_lambda': np.arange(0.5, 10.0, 0.5),
    'scale_pos_weight': np.arange(10.0, 100.0, 1.0),
    'border_count': np.arange(20, 200, 5)
}

# param_random_gb = {
#     'n_estimators': np.arange(100, 130, 5),
#     'max_depth': [7],
#     'learning_rate': np.arange(0.01, 0.03, 0.005),
#     'reg_lambda': np.arange(3.5, 5.0, 0.5),
#     'scale_pos_weight': np.arange(25.0, 30.0, 1.0),
#     'border_count': np.arange(115, 130, 5),
# }
gb = CatBoostClassifier(loss_function='Logloss')
print(gb)

search = RandomizedSearchCV(estimator=gb, param_distributions=param_random_gb, n_iter=30,
                            scoring='recall', cv=2, verbose=100, n_jobs=-1)

search.fit(X, y)

print("Best parameter: ", search.best_params_)
