import pandas as pd

from src.balancing import data_controller
from src.balancing.multiclass_resampling import *
from src.balancing.undersampling import *
from src.balancing.oversampling import *

from src.balancing.utilities import split_data_on_x_y, count_classes_size, train_and_score, train_all_from_dir, \
    resample_and_write_to_csv, resampler_selector


def balance_all_methods(filepath_source):
    resampler_names = ['none', 'ros', 'smotenc', 'rus', 'nearmiss', 'enn', 'renn', 'allknn', 'onesided',
                       'ncr', 'iht', 'globalcs', 'soup']

    for name in resampler_names:
        resampler_selector(name, filepath_source)


if __name__ == '__main__':
    filepath = '../../data/feature_selected_data/imbalance_set.csv'
    data = data_controller.get_categorized_criteo(filepath)

    X_original, y_original = split_data_on_x_y(data)
    print("Classes size:", count_classes_size(y_original))

    balance_all_methods(filepath)

    print("No balancing:")
    train_and_score(X_original, y_original)
    train_all_from_dir('../../data/balanced_csv')
