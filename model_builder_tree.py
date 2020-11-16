from sklearn import tree
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

tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(X_train, y_train)

y_predicted = tree_clf.predict(X_test)

counter = 0
for i in range(len(y_predicted)):
    counter += 1 if y_predicted[i] != y_test[i] else 0

print(tree_clf.score(X_test, y_test))
print(counter)


