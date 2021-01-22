import os

from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

from src.balancing import data_controller, oversampling, undersampling
from src.balancing.data_controller import path_data_dir, path_balanced_csv
import pandas as pd


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
    print(classification_report_imbalanced(y_test, y_pred))
    print_metrics(y_test, y_pred)

    return geometric_mean_score(y_test, y_pred)


def percent_change(original, value):
    return round((value - original) / original * 100, 2)


def resample_and_write_to_csv(obj, X, y, name=None):
    if name is None:
        name = obj.__str__().split("(")[0]

    X_resampled, y_resampled = obj.fit_resample(X, y)
    # print("Classes size:", count_classes_size(y_resampled))

    write_df = X_resampled
    write_df["Sales"] = y_resampled

    filepath = path_balanced_csv + "/" + name + ".csv"

    write_df.to_csv(filepath, index=False)
    print("Balanced:", name, '\n')

    return filepath


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
    for key in ['Sales', 'SalesAmountInEuro', 'time_delay_for_conversion']:  # outcome labels
        if key in features_dict.keys():
            features_dict.pop(key)

    X = data[list(features_dict.keys())]
    y = pd.DataFrame(data['Sales'], columns=["Sales"])

    return X, y


def get_every_nth_element_of_list(L, size_of_sublist):
    if size_of_sublist >= len(L):
        return L
    elif type(len(L) / size_of_sublist) != int:
        size_of_sublist -= 1

    step = int(len(L) / size_of_sublist)
    if step == 0:
        step = 1

    return L[::step]


def resampler_selector(balancing_type, filepath):
    input_data = data_controller.get_feature_selected_data(filepath)
    X, y = split_data_on_x_y(input_data)

    if balancing_type == 'none':
        return filepath, 'Sales', ','
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
    elif balancing_type == 'edt':
        obj = undersampling.edited_nearest_neighbours_optimized()
    elif balancing_type == 'rep_edt':
        obj = undersampling.repeated_edited_nearest_neighbours_optimized()
    elif balancing_type == 'allknn':
        obj = undersampling.allknn_optimized()
    elif balancing_type == 'condensed':
        obj = undersampling.condensed_nearest_neighbours_optimized()
    elif balancing_type == 'onesided':
        obj = undersampling.one_sided_selection_optimized()
    elif balancing_type == 'neighbrhd':
        obj = undersampling.neighbourhood_cleaning_rule_optimized()
    elif balancing_type == 'iht':
        obj = undersampling.instance_hardness_threshold_optimized()
    else:
        print("ERR")
        return None
    return resample_and_write_to_csv(obj, X, y), list(y.columns)[0], ','
