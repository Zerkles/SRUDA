import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time

start = time.time()
data = pd.read_csv('../../data/criteo/balanced.csv', sep=',')
head = 'user_id,partner_id,nb_clicks_1week,product_price,product_age_group,device_type,audience_id,product_gender,click_timestamp,product_id,product_country,product_brand,product_title,product_category6,product_category5,Sales'.split(',')
print(head)

X = data.loc[:, head]
y = data['Sales']

X[np.isnan(X)] = 0

depths = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

min_sample_splits = [x/10 for x in range(1, 9)] + [x for x in range(1, 10)]

max_features = ['auto', 'sqrt', 'log2']

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': depths,
    'min_samples_split': min_sample_splits,
    'min_weight_fraction_leaf': [x/10 for x in range(1, 5)],
    'max_features': max_features,
    'random_state': np.arange(0, 50, 1),
    'n_estimators': np.arange(100, 300, 50),
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample']
}

model = RandomForestClassifier()
print(model)


#search = GridSearchCV(estimator=model, param_grid=params, scoring='recall', cv=2, verbose=100, n_jobs=-1)
search = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='recall', n_iter=1000, cv=3,
                            verbose=100, n_jobs=-1)
search.fit(X, y)

print('============================')
print(search)
print(search.best_params_)
print(search.best_score_)
print(np.sqrt(np.abs(search.best_score_)))
print('Best params:')
print('recall:\t', search.best_params_)


print(time.time() - start)
