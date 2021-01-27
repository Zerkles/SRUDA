import os
import re

from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

from src.balancing.data_controller import DataController
from src.balancing.resampler import Resampler


def train_and_score(X: pd.DataFrame, y: pd.DataFrame):
    X = X.values
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
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

        X_resampled, y_resampled = DataController.split_data_on_x_y(df)
        train_and_score(X_resampled, y_resampled)


def balance_all_methods(filepath_source):
    resampler_names = ['ros', 'smotenc', 'rus', 'nearmiss', 'enn', 'renn', 'allknn', 'onesided',
                       'ncr', 'iht', 'globalcs', 'soup']

    for name in resampler_names:
        res = Resampler(name, filepath_source)
        filepath_destination = re.sub('\w+.csv', '', filepath_source) + '../balanced_csv'
        res.resample_and_write_to_csv(filepath_destination)


if __name__ == '__main__':
    filepath = '../../data/no_price_feature_selected/imbalance_set_no_price.csv'
    data = DataController.read_categorized_criteo(filepath)
    X, y = DataController.split_data_on_x_y(data)
    print("No balancing:")
    train_and_score(X, y)

    balance_all_methods(filepath)
    train_all_from_dir('../../data/balanced_csv')
