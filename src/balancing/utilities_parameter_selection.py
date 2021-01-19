from cmath import sqrt

from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from src.balancing import data_controller
import pandas as pd
import numpy as np

from src.balancing.utilities import resample_and_write_to_csv, split_data_on_x_y, count_classes_size, \
    get_every_nth_element_of_list


def gridsearch_with_graph(resampler_obj, parameters_dist: dict, X, y):
    import matplotlib.pyplot as plt
    import itertools
    # Generate permutations of parameters
    keys, values = zip(*parameters_dist.items())
    permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # print(permutations_dicts)

    print("Parameter ranges:")
    for key in parameters_dist.keys():
        print(f"{key}: {min(parameters_dist[key])}-{max(parameters_dist[key])}")
    print("Iterations:", len(permutations_dict), "\n")

    X = X.values
    y = y.values.ravel()

    gmean_scoring = []
    recall_scoring = []
    for variant in permutations_dict:
        resampler_obj.set_params(**variant)
        X_resampled, y_resampled = resampler_obj.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
        clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        gmean_scoring.append(geometric_mean_score(y_test, y_pred))
        recall_scoring.append(recall_score(y_test, y_pred))
        print("Tested for args:", resampler_obj.get_params())
        # print_metrics(y_test, y_pred)

    # Drawing graph
    fig, axs = plt.subplots(len(parameters_dist.keys()) + 1)
    fig.suptitle(obj.__str__().title().split("(")[0])
    axs[0].plot(range(0, len(gmean_scoring)), gmean_scoring, markersize=7, marker='o')
    axs[0].set(ylabel="Geometric Mean Score")

    axs[-1].set(xlabel="Experiment Number")
    # axs[0].title(obj.__str__().title().split("(")[0])
    # axs[0].xlabel("Experiment Number")
    # axs[0].ylabel('Geometric Mean Score')
    # plt.show()
    for key in parameters_dist.keys():
        param_values = []
        for exp in permutations_dict:
            param_values.append(exp[key])

        ax = axs[list(parameters_dist.keys()).index(key) + 1]
        ax.plot(range(len(param_values)), param_values, markersize=7, marker='o')
        ax.set(ylabel=key)
        # ax.xlabel("Experiment Number")
        # ax.ylabel(key)
        # ax.title(obj.__str__().title().split("(")[0])
        # plt.show()

    # plt.legend(["Geometric Mean Score"]+list(parameters_dist.keys()), loc="upper right")
    plt.show()

    # Displaying top parameters
    parameters_count = 10 + 1
    print("Top parameters:")
    for i in range(1, parameters_count):
        value = max(gmean_scoring)
        index = gmean_scoring.index(value)
        print(i, "Geometric Mean Score", value, "Recall", recall_scoring[index], "for =", permutations_dict[index])
        gmean_scoring[index] = 0
    for i in range(1, parameters_count):
        value = max(recall_scoring)
        index = recall_scoring.index(value)
        print(i, "Recall", value, "Geometric Mean Score", gmean_scoring[index], "for =", permutations_dict[index])
        recall_scoring[index] = 0


def pipeline_randomized_and_grid_search(X, y):
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
    # This is for feature optimization use
    RANDOM_STATE = 0
    data = data_controller.get_feature_selected_data()
    X, y = split_data_on_x_y(data)

    class_size = count_classes_size(y)
    print("Classes size:", class_size)

    max_n_neighbors = class_size[1] - 1
    percent_step = 0.05

    square_root_from_samples_count = int(class_size[1])
    print("Square root from samples count:", square_root_from_samples_count)

    value_dict = {"n_neighbors": get_every_nth_element_of_list(list(range(1, max_n_neighbors)), percent_step),
                  "version": [1, 2, 3],
                  "n_neighbors_ver3": get_every_nth_element_of_list(list(range(1, max_n_neighbors)), percent_step)}

    obj = NearMiss(n_jobs=-1)
    gridsearch_with_graph(obj, value_dict, X, y)
