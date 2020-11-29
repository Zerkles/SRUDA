from collections import Counter

from balancing.multi_imbalanced import compare_multi_class_methods
from balancing.over_sampling import compare_oversampling_methods
from test_train import train_and_score
from balancing.under_sampling import compare_undersampling_methods
from data import data_controller

if __name__ == '__main__':
    samples_count = 100000
    X_original, y_original = data_controller.get_converted_data(samples_count)
    print(sorted(Counter(y_original).items()))

    print("No balancing:")
    score_original = train_and_score(X_original, y_original)
    print("Score:", score_original, "\n")

    compare_oversampling_methods(X_original, y_original, score_original)
    compare_undersampling_methods(X_original, y_original, score_original)
    compare_multi_class_methods(X_original, y_original, score_original)
