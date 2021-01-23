import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
import time


start = time.time()
data = pd.read_csv('../../data/criteo/balanced.csv', sep=',')
head = 'user_id,partner_id,nb_clicks_1week,product_price,product_age_group,device_type,audience_id,product_gender,click_timestamp,product_id,product_country,product_brand,product_title,product_category6,product_category5,Sales'.split(',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()
X[np.isnan(X)] = 0

params = {
    'penalty': ['l2', 'none'],
    'dual': [False],
    'tol': [1/(10**x) for x in range(3, 5)],
    'C': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.0],
    'fit_intercept': [False, True],
    'intercept_scaling': [1.0],
    'solver': ['saga', 'newton-cg'],
    'max_iter': [x for x in range(100, 200, 10)],
    'class_weight': [None, 'balanced']
}

model = LogisticRegression()
print(model)

search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=30,
                            scoring='f1', cv=4, verbose=100, n_jobs=-1)

search.fit(X, y)

print(search)
print(search.best_params_)
print(search.best_score_)
print(np.sqrt(np.abs(search.best_score_)))

print(time.time() - start)
