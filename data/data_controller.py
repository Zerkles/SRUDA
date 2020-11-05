import os
import numpy as np

import pandas as pd

header = ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp',
          'nb_clicks_1week', 'product_price', 'product_age_group', 'device_type', 'audience_id',
          'product_gender', 'product_brand', 'prod_cat_1', 'prod_cat_3', 'prod_cat_4', 'prod_cat_5',
          'prod_cat_6', 'prod_cat_7', 'product_country', 'product_id', 'product_title', 'partner_id',
          'user_id']


def get_raw_data(rows_count: int) -> pd.DataFrame:
    return pd.read_csv(
        filepath_or_buffer=os.path.dirname(
            os.path.abspath(__file__)) + '/Criteo_Conversion_Search/CriteoSearchData',
        nrows=rows_count,
        sep='\t',
        index_col=False,
        low_memory=False,
        names=header
    )


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


def hex_to_float64(data: pd.DataFrame):
    for col in data.columns:
        if str(data[col].dtype) == 'object':
            data[col].apply(lambda x: int(x, 16) >> 64)


def get_converted_data(rows_count):
    # get data
    data = get_raw_data(rows_count)

    # convert hex to float
    # hex_to_float64(data)
    data = data.apply(fun2)

    X = data[header[3:]].values
    y = data['Sales'].values

    X[np.isnan(X)] = 0
    X[np.isinf(X)] = np.finfo(np.float64).max

    return X, y


if __name__ == '__main__':
    df = get_raw_data(1000)
    print(df)
