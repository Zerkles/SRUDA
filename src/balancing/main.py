from collections import Counter

from src.balancing.data_controller import header
from src.balancing.oversampling import balance_all_oversampling
from src.balancing.undersampling import balance_all_undersampling
from src.balancing.utilities import train_and_score, \
    train_and_compare_all

from src.balancing import data_controller

if __name__ == '__main__':
    cores_count = -1  # -1 == all cores
    data = data_controller.get_categorized_data(1000000)
    # data = data_controller.get_pure_data(100)

    X_original = data[header[3:]]
    y_original = data['Sales']

    print("Classes size:")
    print(sorted(Counter(y_original).items()), "\n")

    balance_all_undersampling(X_original, y_original, cores_count)
    balance_all_oversampling(X_original, y_original, cores_count)
    # balance_all_multiclass(X_original, y_original, cores_count)

    print("No balancing:")
    score_original = train_and_score(X_original, y_original, cores_count)
    print("Score:", score_original, "\n")
    train_and_compare_all(score_original, cores_count)
