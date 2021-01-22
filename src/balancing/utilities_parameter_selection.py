import itertools
from cmath import sqrt
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler, TomekLinks, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from src.balancing import data_controller

from src.balancing.utilities import resample_and_write_to_csv, split_data_on_x_y, count_classes_size, \
    get_every_nth_element_of_list, print_metrics, train_and_score


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


def pick_top_parameters(permutations_dict, logs, filepath, parameters_count):
    with open(filepath + '.txt', 'w') as file:  # Use file to refer to the file object
        file.write('Top parameters by Geometric Mean Score:\n')
        gmean_scores = []
        recall_scores = []
        print(logs)

        for key in logs.keys():
            gmean_scores.append(logs[key]["gmean"])
            recall_scores.append(logs[key]["recall"])

        for i in range(1, parameters_count):
            value = max(gmean_scores)
            index = gmean_scores.index(value)
            line = f"{i}. Geometric Mean Score:{value} Recall:{logs[index]['recall']} " \
                   f"for params:{permutations_dict[index]} with classes_size:{logs[index]['classes_size']}"
            print(line)
            file.write(line + '\n')
            gmean_scores[index] = 0
        file.write('----------\n')
        file.write('Top parameters by Recall:\n')
        for i in range(1, parameters_count):
            value = max(recall_scores)
            index = recall_scores.index(value)
            line = f"{i}. Recall:{value} Geometric Mean Score:{logs[index]['gmean']} " \
                   f"for params:{permutations_dict[index]} with classes_size:{logs[index]['classes_size']}"
            print(line)
            file.write(line + '\n')
            recall_scores[index] = 0
        file.write('----------\n')
        for key in logs.keys():
            file.write(f"{key}. {logs[key]}\n")


def draw_plots(parameters_dist, permutations_dict, logs, algorithm_name, filepath):
    gmean_scores = []
    for key in logs.keys():
        gmean_scores.append(logs[key]['gmean'])

    plot_colors = ['blue', 'green', 'orange', 'violet', 'red', 'brown', 'pink']
    fig, axs = plt.subplots(len(parameters_dist.keys()) + 1, **{"figsize": (6.5, 6.5)})
    fig.suptitle(algorithm_name)
    axs[-1].set(xlabel="Experiment Number")

    axs[0].plot(range(1, len(gmean_scores) + 1), gmean_scores, plot_colors[0], markersize=5, marker='o')
    axs[0].set(ylabel="Geometric Mean Score")
    axs[0].grid(True)

    for key in parameters_dist.keys():
        param_values = []
        for exp in permutations_dict:
            param_values.append(exp[key])

        plot_index = list(parameters_dist.keys()).index(key) + 1
        ax = axs[plot_index]
        ax.plot(range(1, len(param_values) + 1), param_values, plot_colors[plot_index], markersize=5, marker='o')
        ax.set(ylabel=key)
        ax.grid(True)

    # plt.legend(["Geometric Mean Score"]+list(parameters_dist.keys()), loc="upper right")
    plt.savefig(filepath + ".png")
    plt.show()


def gridsearch_with_plot(resampler_obj, parameters_dist: dict, X, y):
    # Generate permutations of parameters
    keys, values = zip(*parameters_dist.items())
    parameter_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Estimate parameter ranges
    parameter_ranges = {}
    for key in parameters_dist.keys():
        parameter_ranges.update({key: [min(parameters_dist[key]), max(parameters_dist[key])]})
    print("Parameter ranges:", parameter_ranges)
    print("Iterations:", len(parameter_permutations), "\n")

    algorithm_name = obj.__str__().title().split("(")[0]
    filepath = "../../plots/" + algorithm_name + str(parameter_ranges)

    X = X.values
    y = y.values.ravel()

    logs = {}
    for variant in parameter_permutations:
        resampler_obj.set_params(**variant)
        X_resampled, y_resampled = resampler_obj.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
        clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
        # clf = CatBoostClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        index = parameter_permutations.index(variant)
        logs.update({index: {"params": resampler_obj.get_params(), "gmean": geometric_mean_score(y_test, y_pred),
                             "recall": recall_score(y_test, y_pred),
                             "classes_size": count_classes_size(y_resampled),
                             "roc_auc": roc_auc_score(y_test, y_pred)}})

        # print(f"{index + 1}.", "Tested for args:", logs[index]['params'])
        print(index + 1, logs[index])

        # print_metrics(y_test, y_pred)

    # Drawing graph
    draw_plots(parameters_dist, parameter_permutations, logs, algorithm_name, filepath)

    # Displaying top parameters
    pick_top_parameters(parameter_permutations, logs, filepath, 10 + 1)


if __name__ == "__main__":
    # This is for feature optimization use
    data = data_controller.get_feature_selected_data(data_controller.path_feature_selected)
    # data = data_controller.get_categorized_data(100000)
    X, y = split_data_on_x_y(data)

    class_size = count_classes_size(y)
    print("Classes size:", class_size)

    RANDOM_STATE = 0
    max_n_neighbors = class_size[1] - 1

    # square_root_from_samples_count = int(class_size[1])
    # print("Square root from samples count:", square_root_from_samples_count)

    value_dict = {"n_neighbors": get_every_nth_element_of_list(list(range(2, 1226)), 100),
                  "kind_sel": ['all']
                  }
    # {'gmean': 0.9986377090714849, 'recall': 0.9992784992784993, 'classes_size': {0: 4983, 1: 4630}, 'params': {'allow_minority': True, 'kind_sel': 'all', 'n_jobs': -1, 'n_neighbors': 422, 'sampling_strategy': 'auto'}}
    # Geometric Mean Score:0.9666639562161194 Recall:0.9539800995024875 for params:{'n_neighbors': 12, 'kind_sel': 'all'} with classes_size:{0: 9884, 1: 10760}
    obj = AllKNN(sampling_strategy='auto', n_jobs=-1, allow_minority=True)
    gridsearch_with_plot(obj, value_dict, X, y)
