import os

from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from src.balancing import data_controller
from src.balancing.data_controller import path_data_dir
import pandas as pd
import numpy as np

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


def resample_and_write_to_csv(obj, X, y, name):
    X_resampled, y_resampled = obj.fit_resample(X, y)
    check_if_new_categorical_features_generated(X, X_resampled)

    # write_df = X_resampled.append(y_resampled)
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

    if not all(elem in X["product_id"].unique() for elem in X_resampled["product_id"].unique()):
        print("WARNING! New categorical features are generated!")


def feature_graph_generator(resampler_obj, parameters_dist: dict, algorithm_name, X, y):
    import matplotlib.pyplot as plt
    import itertools
    keys, values = zip(*parameters_dist.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # print(permutations_dicts)

    print("Parameter ranges:")
    for key in parameters_dist.keys():
        print(f"{key}: {min(parameters_dist[key])}-{max(parameters_dist[key])}")
    print("Iterations:", len(permutations_dicts))

    gmean_scoring = []
    recall_scoring = []
    for variant in permutations_dicts:
        resampler_obj.set_params(**variant)
        X_resampled, y_resampled = resampler_obj.fit_resample(X, y)
        # check_if_new_categorical_features_generated(X, X_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
        clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        gmean_scoring.append(geometric_mean_score(y_test, y_pred))
        recall_scoring.append(recall_score(y_test, y_pred))
        print("Tested for args:", resampler_obj.get_params())
        # print_metrics(y_test, y_pred)

    plt.plot(range(1, len(gmean_scoring) + 1), gmean_scoring, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=7)
    plt.title(algorithm_name)
    plt.xlabel(str(list(parameters_dist.keys())))
    plt.ylabel('Geometric Mean Score')
    plt.show()

    parameters_count = 20 +1
    print("Top parameters:")
    for i in range(1, parameters_count):
        value = max(gmean_scoring)
        index = recall_scoring.index(value)
        print(i, "Score", value, "Recall", recall_scoring[index], "for =", permutations_dicts[index])
        gmean_scoring[index] = 0
    for i in range(1, parameters_count):
        value = max(recall_scoring)
        index = recall_scoring.index(value)
        print(i, "Recall", value, "Score", gmean_scoring[index], "for =", permutations_dicts[index])
        recall_scoring[index] = 0


def randomized_search(X, y):
    X = X.values
    y = y.values.ravel()

    max_neighbors_count = 2714
    print(max_neighbors_count)
    gmean_scorer = make_scorer(geometric_mean_score)

    resampler_name = 'nearmiss__'
    resampler = NearMiss(n_jobs=-1)
    classifier = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
    param_dist = {resampler_name + "n_neighbors": list(range(1, max_neighbors_count)),
                  # resampler_name + "n_neighbors_ver3": list(range(1, max_neighbors_count)),
                  resampler_name + "version": [1, 2, 3]}

    pipeline = Pipeline([('nearmiss', resampler), ('RandomForestClassifier', classifier)])
    # print(pipeline.get_params().keys())

    random_search = RandomizedSearchCV(pipeline, n_jobs=-1, param_distributions=param_dist, n_iter=30, cv=5,
                                       random_state=0, scoring=gmean_scorer)
    random_search.fit(X, y)
    print(random_search.best_params_)
    print(random_search.best_score_)

    grid_search = GridSearchCV(pipeline, n_jobs=-1, param_grid=param_dist, cv=3, scoring=gmean_scorer)

    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)


if __name__ == "__main__":
    from cmath import sqrt

    # This is for feature optimization use
    data = data_controller.get_categorized_data(100000)

    X = data[list(data.columns)[3:]]
    y = pd.DataFrame(data['Sales'], columns=["Sales"])

    X = X.values
    y = y.values.ravel()

    obj_list = []
    max_n_neighbors = np.count_nonzero(y == 1) - 1
    # max_n_neighbors = 2139

    square_root_from_samples_count = int(sqrt(y.shape[0]).real)
    print("Square root from samples count:", square_root_from_samples_count)

    resampler_name = "NearMiss for "  # + str(int(y.count())) + " samples"

    percent_step = 0.05
    value_dict = {"n_neighbors": list(range(1, max_n_neighbors, int(percent_step * max_n_neighbors))),
                  "version": [3],
                  "n_neighbors_ver3": list(range(1, max_n_neighbors, int(percent_step * max_n_neighbors)))}
    feature_graph_generator(NearMiss(n_jobs=-1), value_dict, resampler_name, X, y)
