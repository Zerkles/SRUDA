import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier

data = pd.read_csv('../../data/criteo/balanced.csv', sep=',')
head = 'user_id,partner_id,nb_clicks_1week,product_price,product_age_group,device_type,audience_id,product_gender,click_timestamp,product_id,product_country,product_brand,product_title,product_category6,product_category5,Sales'.split(',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
X[np.isnan(X)] = 0

house_dmatrix = xg.DMatrix(data=X, label=y)

param_random_gb = {
    'n_estimators': np.arange(100, 130, 5),
    'max_depth': [7],
    'learning_rate': np.arange(0.01, 0.03, 0.005),
    'reg_lambda': np.arange(3.5, 5.0, 0.5),
    'scale_pos_weight': np.arange(25.0, 30.0, 1.0),
    'border_count': np.arange(115, 130, 5),
}
gb = CatBoostClassifier(loss_function='Logloss')
print(gb)

search = RandomizedSearchCV(estimator=gb, param_distributions=param_random_gb, n_iter=5,
                            scoring='f1', cv=2, verbose=100, n_jobs=-1)

search.fit(X, y)

print("Best parameter: ", search.best_params_)
