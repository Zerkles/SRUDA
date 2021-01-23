import pickle
import pandas as pd

header = ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion', 'click_timestamp',
          'nb_clicks_1week', 'product_price', 'product_age_group', 'device_type', 'audience_id',
          'product_gender', 'product_brand', 'product_category1', 'product_category2', 'product_category3',
          'product_category4', 'product_category5', 'product_category6', 'product_category7', 'product_country',
          'product_id', 'product_title', 'partner_id', 'user_id']


def get_raw_criteo(filepath: str, rows_count: int = -1):
    if rows_count != -1:
        return pd.read_csv(
            filepath_or_buffer=filepath,
            nrows=rows_count,
            sep='\t',
            index_col=False,
            low_memory=False,
            names=header
        )
    else:
        return pd.read_csv(
            filepath_or_buffer=filepath,
            sep='\t',
            index_col=False,
            # low_memory=False,
            names=header
        )


def get_categorized_criteo(filepath: str, rows_count: int = -1) -> pd.DataFrame:
    if rows_count != -1:
        return pd.read_csv(
            filepath_or_buffer=filepath,
            nrows=rows_count,
            sep=',',
            index_col=False,
            low_memory=False,
        )
    return pd.read_csv(
        filepath_or_buffer=filepath,
        sep=',',
        index_col=False,
        low_memory=False,
    )


def encode_data(filepath_criteo: str, filepath_categorized_criteo: str, filepath_label_encoder: str):
    from sklearn.preprocessing import LabelEncoder

    df = get_raw_criteo(filepath_criteo)

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

    for title in df["product_category7"].unique():
        if type(title) != str:
            print(type(title), title)
    df["product_title"] = df["product_title"].astype(str)
    df["product_category7"] = df["product_category7"].astype(str)

    for key in label_encoder_dict.keys():
        label_encoder_dict[key].fit(df[key])
        df[key] = label_encoder_dict[key].transform(df[key])

    df.to_csv(filepath_categorized_criteo, index=False)
    pickle.dump(label_encoder_dict, open(filepath_label_encoder, "wb"))


def decode_data(df: pd.DataFrame, filepath_label_encoder: str) -> pd.DataFrame:
    label_encoder_dict = pickle.load(open(filepath_label_encoder, "rb"))

    for key in label_encoder_dict.keys():
        df[key] = label_encoder_dict[key].inverse_transform(df[key])

    return df
