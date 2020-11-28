#!/home/kwitnoncy/anaconda3/envs/inz/bin/python
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from collections import Counter
import numpy as np
import pandas as pd

data = pd.read_csv('data_100k.csv', sep='\t')
#data = pd.read_csv('whole_formated.csv', sep='\t')
print(data.columns)

X = data.loc[:, ['click_timestamp', 'nb_clicks_1week', 'product_price',
     'product_age_group', 'device_type', 'audience_id', 'product_gender',
     'product_brand', 'prod_cat_1', 'prod_cat_3', 'prod_cat_4', 'prod_cat_5',
     'prod_cat_6', 'prod_cat_7', 'product_country', 'product_id',
     'product_title', 'partner_id', 'user_id']]
y = data['Sales']

X = X.to_numpy()
y = y.to_numpy()

# do some stuff with data. Maybe check what parameters are relevant
X[np.isnan(X)] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
print("pre balanced:", sorted(Counter(y_test).items()))
print("pre balanced:", sorted(Counter(y_train).items()))

clf = svm.SVC()
clf = svm.SVR(cache_size=7000)
#X, y = [[0, 0], [1, 1]], [0, 1]
print(clf.fit(X_train, y_train))

print('ok')

print(clf.score(X_test, y_test))

