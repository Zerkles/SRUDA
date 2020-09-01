from imblearn.over_sampling import ADASYN, RandomOverSampler
from collections import Counter
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

header_X = [
    'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp',
    'nb_clicks_1week', 'product_price', 'product_age_group', 'device_type',
    'audience_id', 'product_gender', 'product_brand', 'prod_cat_1',
    'prod_cat_3', 'prod_cat_4', 'prod_cat_5', 'prod_cat_6', 'prod_cat_7',
    'product_country', 'product_id', 'product_title', 'partner_id', 'user_id'
]

data = pd.read_csv(
    filepath_or_buffer='mini_formated.csv',
    sep='\t',
    low_memory=False).tail(500000)

y = data['Sales'].to_numpy()
X = data[header_X].to_numpy()

X[np.isnan(X)] = 0
X[np.isinf(X)] = np.finfo(np.float64).max

print("pre balanced:", sorted(Counter(y).items()))


X_resampled, y_resampled = ADASYN().fit_resample(X, y)
print("post balanced (ADASYN):", sorted(Counter(y_resampled).items()))
# clf_adasyn = LinearSVC().fit(X_resampled, y_resampled)
