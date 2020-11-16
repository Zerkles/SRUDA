from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pandas as pd


# open file with balanced and unbalanced data
#data = pd.read_csv('data_100k.csv', sep='\t')
data = pd.read_csv('whole_formated.csv', sep='\t')
print(data.columns)
#X = data.loc[:, data.columns != ['Sales', 'SalesAmountInEuro',
#                                 'time_delay_for_conversion']].to_numpy()
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


# learn model

mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000)
mlp_clf.fit(X_train, y_train)

# run test and validate model
counter_false = 0
#for row in zip(X_test, y_test):
#    if mlp_clf.predict(row[0]) != row[1]:
#        counter_false += 1

# create score's and save them
print("samples count: ", len(y_test))
print("counter_false: ", counter_false)
print("score: ", mlp_clf.score(X_test, y_test))
