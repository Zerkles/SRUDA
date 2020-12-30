import os
import pickle

import numpy as np

import pandas as pd

header = ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp',
          'nb_clicks_1week', 'product_price', 'product_age_group', 'device_type', 'audience_id',
          'product_gender', 'product_brand', 'product_category1', 'product_category2', 'product_category3',
          'product_category4', 'product_category5', 'product_category6', 'product_category7', 'product_country',
          'product_id', 'product_title', 'partner_id', 'user_id']

# path_data_dir = str(os.getcwd() + '/data')
path_data_dir = '././data'
categorized_criteo_filename = 'CriteoSearchDataCategorized.csv'
features_criteo = 'matrix_feature.csv'
path_data_original_criteo = path_data_dir + '/criteo/' + 'CriteoSearchData'
path_categorized_criteo = path_data_dir + '/criteo/' + categorized_criteo_filename
path_labelEncoderDict = path_data_dir + '/criteo/' + "LabelEncoderDict.pickle"


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


def get_pure_data(rows_count=-1):  # from raw data
    if rows_count != -1:
        data = pd.read_csv(
            filepath_or_buffer=path_data_original_criteo,
            nrows=rows_count,
            sep='\t',
            index_col=False,
            low_memory=False,
            names=header
        )
    else:
        data = pd.read_csv(
            filepath_or_buffer=path_data_original_criteo,
            sep='\t',
            index_col=False,
            low_memory=False,
            names=header
        )
    # convert hex to float
    hex_to_float64(data)
    data = data.apply(fun2)

    # X = data[header[3:]].values
    # y = data['Sales'].values

    data[np.isnan(data)] = 0
    data[np.isinf(data)] = np.finfo(np.float64).max

    return data


def get_categorized_data(rows_count: int = -1) -> pd.DataFrame:
    if rows_count != -1:
        return pd.read_csv(
            filepath_or_buffer=path_categorized_criteo,
            nrows=rows_count,
            sep=',',
            index_col=False,
            low_memory=False,
        )
    return pd.read_csv(
        filepath_or_buffer=path_categorized_criteo,
        sep=',',
        index_col=False,
        low_memory=False,
    )


def label_data():
    from sklearn.preprocessing import LabelEncoder

    df = get_pure_data()

    label_encoder_dict = {'product_age_group': LabelEncoder(),
                          'device_type': LabelEncoder(),
                          'audience_id': LabelEncoder(),
                          'product_gender': LabelEncoder(),
                          'product_brand': LabelEncoder(),
                          'product_category1': LabelEncoder(),
                          'product_category2': LabelEncoder(),
                          'product_category3': LabelEncoder(),
                          'product_category4': LabelEncoder(),
                          'product_category5': LabelEncoder(),
                          'product_category6': LabelEncoder(),
                          'product_category7': LabelEncoder(),
                          'product_country': LabelEncoder(),
                          'product_id': LabelEncoder(),
                          'product_title': LabelEncoder(),
                          'partner_id': LabelEncoder(),
                          'user_id': LabelEncoder()}

    print("Dataframe loaded!")

    print(df["product_title"].unique())
    df["product_title"] = df["product_title"].astype(str)

    for key in label_encoder_dict.keys():
        print("Labeling:", key)
        label_encoder_dict[key].fit(df[key])
        df[key] = label_encoder_dict[key].transform(df[key])

    df.to_csv(categorized_criteo_filename)
    pickle.dump(label_encoder_dict, open(path_labelEncoderDict, "wb"))


def delabel_data(df: pd.DataFrame) -> pd.DataFrame:
    label_encoder_dict = pickle.load(open(path_labelEncoderDict, "rb"))

    for key in label_encoder_dict.keys():
        print("Delabeling:", key)
        df[key] = label_encoder_dict[key].inverse_transform(df[key])

    return df


if __name__ == '__main__':
    label_data()
