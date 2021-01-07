#!/home/kwitnoncy/anaconda3/envs/inz/bin/python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import numpy as np
import pandas as pd


# open file with balanced and unbalanced data
data = pd.read_csv('../../data/criteo/criteo_40k.csv', sep='\t')
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

X_train, X_test = scale(X_train), scale(X_test)
# learn model
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

etas = [0.3, 0.25, 0.2, 0.15, 0.1]
max_depths = [6, 10, 15, 20, 25]

# model = XGBClassifier(eta=0.1, max_depth=3)
model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=10)
model.fit(X_train, y_train)

print(model)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %f" % accuracy)

y_pred = model.predict(X_train)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_train, predictions)
print("Accuracy: %f" % (accuracy))
print('report:')
print(classification_report(y_true=y_train, y_pred=y_pred))

