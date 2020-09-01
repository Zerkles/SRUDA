from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np


def fun(x):
    if type(x) is int or type(x) is float:
        return x
    try:
        return np.float64((int(x, 16) >> 64))
    except TypeError as e:
        print(x)
        print(e)


def fun2(x):
    return x.apply(fun)


header = [
    'Sales', 'SalesAmountInEuro', 'time_delay_for_conversion',
    'click_timestamp', 'nb_clicks_1week', 'product_price', 'product_age_group',
    'device_type', 'audience_id', 'product_gender', 'product_brand',
    'prod_cat_1', 'prod_cat_3', 'prod_cat_4', 'prod_cat_5', 'prod_cat_6',
    'prod_cat_7', 'product_country', 'product_id', 'product_title',
    'partner_id', 'user_id'
]

header_X = [
    'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp',
    'nb_clicks_1week', 'product_price', 'product_age_group', 'device_type',
    'audience_id', 'product_gender', 'product_brand', 'prod_cat_1',
    'prod_cat_3', 'prod_cat_4', 'prod_cat_5', 'prod_cat_6', 'prod_cat_7',
    'product_country', 'product_id', 'product_title', 'partner_id', 'user_id'
]

print("started")

data = pd.read_csv(
    filepath_or_buffer='data/Criteo_Conversion_Search/CriteoSearchData_copy_edited',
    #filepath_or_buffer='minimini',
    sep='\t',
    index_col=False,
    names=header,
    low_memory=False)

print('Opened file')

data = data.apply(fun2)

print('Converted hex to float')

data.to_csv("whole_formated.csv", sep='\t', index=False)
