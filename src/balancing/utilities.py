import os

from imblearn.metrics import geometric_mean_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

from src.balancing.data_controller import path_data_dir
import pandas as pd

path_balanced_csv = path_data_dir + "/balanced_csv"


def train_and_score(X, y):
    if "Sales" in list(X.columns):
        print(X.columns, y.columns)

    X = X.values
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # print("Classification Report Imbalanced:\n", classification_report_imbalanced(y_test, y_pred))
    # print(classification_report_imbalanced(y_test, y_pred))
    print_metrics(y_test, y_pred)

    return geometric_mean_score(y_test, y_pred)


def percent_change(original, value):
    return round((value - original) / original * 100, 2)


def resample_and_write_to_csv(obj, X, y, name=None):
    if name is None:
        name = obj.__str__()

    X_resampled, y_resampled = obj.fit_resample(X, y)

    # print("Classes size:", count_classes_size(y_resampled))

    write_df = X_resampled.copy()
    write_df["Sales"] = y_resampled
    write_df.to_csv(path_balanced_csv + "/" + name + ".csv", index=False)
    print("Balanced:", name, '\n')

    return X_resampled, y_resampled


def train_all_from_csv():
    for filename in os.listdir(path_balanced_csv):
        if filename == ".gitignore":
            continue
        path_file = path_balanced_csv + '/' + filename
        print("Training:", filename[:-4])
        df = pd.read_csv(
            filepath_or_buffer=path_file,
            sep=',',
            index_col=0,
            low_memory=False)

        df_columns = list(df.columns)
        df_columns.remove("Sales")
        X_resampled = df[df_columns]
        y_resampled = df["Sales"]
        train_and_score(X_resampled, y_resampled)


def print_metrics(y_true, y_pred):
    from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support, sensitivity_score, \
        specificity_score, geometric_mean_score, make_index_balanced_accuracy

    print("Geometric Mean Score:", geometric_mean_score(y_true, y_pred))
    print("Precision Score:", precision_score(y_true, y_pred))
    print("Sensitivity Core (Recall):", sensitivity_score(y_true, y_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_true, y_pred))

    # print("Specificity Score:", specificity_score(y_true, y_pred))
    # print("Make Index Balanced Accuracy:", make_index_balanced_accuracy(y_true, y_pred))

    # print("Classification Report Imbalanced:\n", classification_report_imbalanced(y_true, y_pred))
    # print("Sensitivity Specifity Support:", sensitivity_specificity_support(y_true, y_pred))

    print("\n")


def check_if_new_categorical_features_generated(X, X_resampled):
    # print("Unique product id count in original data:", len(X["product_id"].unique()),
    #       "Unique product id count in resampled data:", len(X_resampled["product_id"].unique()))

    if not all(elem in np.unique(X) for elem in np.unique(X_resampled)):
        print("WARNING! New categorical features are generated!")


def count_classes_size(y: pd.DataFrame):
    # class_1_size = np.count_nonzero(y == 1)
    # class_0_size = len(y) - class_1_size

    class_1_size = len(y.loc[y["Sales"] == 1])
    class_0_size = len(y) - class_1_size

    return {0: class_0_size, 1: class_1_size}


def split_data_on_x_y(data: pd.DataFrame):
    features_dict = dict(zip(list(data.columns), range(len(data.columns))))
    for key in ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion']:  # outcome labels
        if key in features_dict.keys():
            features_dict.pop(key)

    X = data[list(features_dict.keys())]
    y = pd.DataFrame(data['Sales'], columns=["Sales"])

    return X, y


def get_every_nth_element_of_list(L, percent_step):
    step = int(percent_step * len(L))
    if step == 0:
        step = 1

    return L[::step]
