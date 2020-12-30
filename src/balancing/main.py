from collections import Counter

import pandas as pd
from src.balancing.data_controller import header
from src.balancing.multiclass import balance_all_multiclass
from src.balancing.utilities import train_and_score, \
    train_and_compare_all

from src.balancing import data_controller

if __name__ == '__main__':
    cores_count = -1  # -1 == all cores
    data = data_controller.get_categorized_data(50000)
    # data = data_controller.get_converted_data(100)

    X_original = data[header[3:]]
    y_original = pd.DataFrame(data['Sales'], columns=["Sales"])
    y_original.rename(columns={"": "Sales"}, inplace=True)

    print("Classes size:")
    print(sorted(Counter(y_original).items()), "\n")

    # balance_all_undersampling(X_original, y_original, cores_count)
    # balance_all_oversampling(X_original, y_original, cores_count)
    balance_all_multiclass(X_original, y_original)

    print("No balancing:")
    score_original = train_and_score(X_original, y_original, cores_count)
    print("Score:", score_original, "\n")
    train_and_compare_all(score_original, cores_count)
