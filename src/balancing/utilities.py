import os
import re

from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

from src.balancing import data_controller, oversampling, undersampling, multiclass_resampling
import pandas as pd


def train_and_score(X: pd.DataFrame, y: pd.DataFrame):
    X = X.values
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Classes size:', count_classes_size(y))
    print(classification_report_imbalanced(y_test, y_pred))

    return geometric_mean_score(y_test, y_pred)


def train_all_from_dir(filepath_balanced_dir: str):
    for filename in os.listdir(filepath_balanced_dir):
        if not filename.endswith('.csv'):
            continue

        path_file = filepath_balanced_dir + '/' + filename
        print("Training:", filename.replace('.csv', ''))
        df = pd.read_csv(
            filepath_or_buffer=path_file,
            sep=',',
            index_col=0,
            low_memory=False)

        X_resampled, y_resampled = split_data_on_x_y(df)
        train_and_score(X_resampled, y_resampled)


def resample_and_write_to_csv(obj, X, y, filepath_destination: str, name: str = None):
    if name is None:
        name = obj.__str__().split("(")[0]

    X_resampled, y_resampled = obj.fit_resample(X, y)

    balanced_df = X_resampled
    balanced_df["Sales"] = y_resampled

    filepath_balanced = f"{filepath_destination}/{name}.csv"

    balanced_df.to_csv(filepath_balanced, index=False)
    print("Balanced:", name)

    return filepath_balanced


def check_if_new_categorical_features_generated(X: np.ndarray, X_resampled: np.ndarray):
    # print("Unique product id count in original data:", len(X["product_id"].unique()),
    #       "Unique product id count in resampled data:", len(X_resampled["product_id"].unique()))

    if not all(elem in np.unique(X) for elem in np.unique(X_resampled)):
        print("WARNING! New categorical features are generated!")


def count_classes_size(y):
    if type(y) == pd.DataFrame:
        class_1_size = len(y.loc[y["Sales"] == 1])
        class_0_size = len(y) - class_1_size
    else:
        class_1_size = np.count_nonzero(y == 1)
        class_0_size = len(y) - class_1_size

    return {0: class_0_size, 1: class_1_size}


def split_data_on_x_y(data: pd.DataFrame):
    features_dict = dict(zip(list(data.columns), range(len(data.columns))))
    for key in ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion']:  # remove outcome labels
        if key in features_dict.keys():
            features_dict.pop(key)

    X = data[list(features_dict.keys())]
    y = pd.DataFrame(data['Sales'], columns=["Sales"])

    return X, y


def get_n_elements_of_list(original_list: list, n_elements: int):
    if n_elements >= len(original_list):
        return original_list
    elif type(len(original_list) / n_elements) != int:
        n_elements -= 1

    step = int(len(original_list) / n_elements)
    if step == 0:
        step = 1

    return original_list[::step]


def resampler_selector(balancing_type: str, filepath_source: str):
    input_data = data_controller.get_categorized_criteo(filepath_source)
    X, y = split_data_on_x_y(input_data)

    filepath_destination = re.sub('\w+.csv', '', filepath_source) + '../balanced_csv'

    if balancing_type == 'none':
        return filepath_source, 'Sales', ','
    elif balancing_type == 'ros':
        obj = oversampling.random_over_sampler_optimized()
    elif balancing_type == 'smotenc':
        if input_data.shape[0] > 10000:  # it just cant handle more than 10k samples because of ram
            X = X.head(10000)
            y = y.head(10000)
        obj = oversampling.smotenc_optimized(X)
    elif balancing_type == 'rus':
        obj = undersampling.random_under_sampler_optimized()
    elif balancing_type == 'nearmiss':
        obj = undersampling.nearmiss_optimized()
    elif balancing_type == 'enn':
        obj = undersampling.edited_nearest_neighbours_optimized()
    elif balancing_type == 'renn':
        obj = undersampling.repeated_edited_nearest_neighbours_optimized()
    elif balancing_type == 'allknn':
        obj = undersampling.allknn_optimized()
    elif balancing_type == 'onesided':
        obj = undersampling.one_sided_selection_optimized()
    elif balancing_type == 'ncr':
        obj = undersampling.neighbourhood_cleaning_rule_optimized()
    elif balancing_type == 'iht':
        obj = undersampling.instance_hardness_threshold_optimized()
    elif balancing_type == 'globalcs':
        obj = multiclass_resampling.global_cs_optimized()
    elif balancing_type == 'soup':
        obj = multiclass_resampling.soup_optimized()
    else:
        print("ERR")
        return None
    return resample_and_write_to_csv(obj, X, y, filepath_destination), list(y.columns)[0], ','
