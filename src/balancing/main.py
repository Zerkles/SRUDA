import pandas as pd

from src.balancing.multiclass import balance_all_multiclass
from src.balancing.oversampling import balance_all_oversampling
from src.balancing.undersampling import balance_all_undersampling
from src.balancing.utilities import train_and_score, train_all_from_csv, feature_optimizer

from src.balancing import data_controller

if __name__ == '__main__':
    data = data_controller.get_categorized_data(1000)
    # data = data_controller.get_converted_data(100)

    X_original = data[list(data.columns)[3:]]
    y_original = pd.DataFrame(data['Sales'], columns=["Sales"])

    print("Classes size:")
    print("1:", len(y_original.loc[y_original["Sales"] == 1]), "0:",
          y_original.shape[0] - len(y_original.loc[y_original["Sales"] == 1]))

    balance_all_undersampling(X_original, y_original)
    balance_all_oversampling(X_original, y_original)
    balance_all_multiclass(X_original, y_original)

    print("No balancing:")
    train_and_score(X_original, y_original)
    train_all_from_csv()
