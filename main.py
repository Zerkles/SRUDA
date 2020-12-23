from collections import Counter

from data.data_controller import header
from utilities import train_and_score, balance_to_csv_all_undersampling, compare_multi_class_methods, \
    balance_to_csv_all_oversampling, balance_train_comparsion

from data import data_controller

if __name__ == '__main__':
    cores_count = -1  # -1 == all cores

    data = data_controller.get_categorized_data()
    # data = data_controller.get_categorized_data(100)
    X_original = data[header[3:]]
    y_original = data['Sales']

    print("Classes size:")
    print(sorted(Counter(y_original).items()), "\n")

    balance_to_csv_all_undersampling(X_original, y_original, cores_count)
    # balance_to_csv_all_oversampling(X_original, y_original, cores_count)
    # compare_multi_class_methods(X_original, y_original, score_original,cores_count)

    print("No balancing:")
    score_original = train_and_score(X_original, y_original, cores_count)
    print("Score:", score_original, "\n")
    balance_train_comparsion(score_original, cores_count)
