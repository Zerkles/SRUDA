import pandas as pd

from src.balancing.multiclass import balance_all_multiclass
from src.balancing.oversampling import balance_all_oversampling
from src.balancing.undersampling import balance_all_undersampling
from src.balancing.utilities import train_and_score, train_all_from_csv, count_classes_size, split_data_on_x_y

from src.balancing import data_controller

if __name__ == '__main__':
    # data = data_controller.get_categorized_data(1000)
    # data = data_controller.get_converted_data(100)
    data = data_controller.get_feature_selected_data(filepath)

    X_original, y_original = split_data_on_x_y(data)
    print("Classes size:", count_classes_size(y_original))

    balance_all_undersampling(X_original, y_original)
    # balance_all_oversampling(X_original, y_original)
    # balance_all_multiclass(X_original, y_original)

    print("No balancing:")
    train_and_score(X_original, y_original)
    train_all_from_csv()
