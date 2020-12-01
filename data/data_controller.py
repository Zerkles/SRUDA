import os
import pickle

import numpy as np

import pandas as pd
from partd import pandas

header = ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp',
          'nb_clicks_1week', 'product_price', 'product_age_group', 'device_type', 'audience_id',
          'product_gender', 'product_brand', 'prod_cat_1', 'prod_cat_3', 'prod_cat_4', 'prod_cat_5',
          'prod_cat_6', 'prod_cat_7', 'product_country', 'product_id', 'product_title', 'partner_id',
          'user_id']

header2 = ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp',
          'nb_clicks_1week', 'product_price','product_age_group','device_type','audience_id',
                  'product_gender', 'product_brand','prod_category1','prod_category2',
                  'prod_category3','prod_category4','prod_category5',
                  'prod_category6','prod_category7','product_country',
                  'product_id','product_title','partner_id','user_id']

data_dir_path = os.path.dirname(os.path.abspath(__file__))


def get_raw_data(rows_count: int) -> pd.DataFrame:
    return pd.read_csv(
        filepath_or_buffer=data_dir_path + '/Criteo_Conversion_Search/CriteoSearchData',
        nrows=rows_count,
        sep='\t',
        index_col=False,
        low_memory=False,
        names=header
    )


def get_raw_data_all() -> pd.DataFrame:
    return pd.read_csv(
        filepath_or_buffer=data_dir_path + '/Criteo_Conversion_Search/CriteoSearchData',
        sep='\t',
        index_col=False,
        low_memory=False,
        names=header
    )


# def get_converted_data_all():
#
#     labelEncoders = pickle.load(open(data_dir_path+"/criteo/lablencoder.pickle", "rb"))
#     print(labelEncoders)
#     df = pd.DataFrame.from_dict(labelEncoders)
#     print(type(df))
#     print(df)

def get_categorized_data(rows_count: int):
    df = pd.read_parquet(
        path=data_dir_path + '/criteo/NEW/criteoCategorized.parquet',
    )

    return df.drop(df.columns[0], axis=1)


def get_categorized_data_all():
    return pd.read_csv(
        filepath_or_buffer=data_dir_path + '/criteo/csv/criteoCategorized.csv',
        sep=',',
        index_col=False,
        low_memory=False,
        # names=header
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
    # data = get_raw_data(rows_count)
    data = get_categorized_data(rows_count)

    # convert hex to float
    # hex_to_float64(data)
    # data = data.apply(fun2)

    X = data[header2[3:]].values
    y = data['Sales'].values

    X[np.isnan(X)] = 0
    X[np.isinf(X)] = np.finfo(np.float64).max

    return X, y


if __name__ == '__main__':
    # df = get_raw_data_all(10000000)
    # print(df)
    print(get_categorized_data(1000))
