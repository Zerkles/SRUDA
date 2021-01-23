import itertools
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, OneSidedSelection
from multi_imbalance.resampling.global_cs import GlobalCS
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import recall_score, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from src.balancing import data_controller

from src.balancing.utilities import split_data_on_x_y, count_classes_size, \
    get_n_elements_of_list


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


def top_n_params(logs: dict, metric_name: str, n_params: int):
    values = []
    lines_to_file = []
    for key in logs.keys():
        values.append(logs[key][metric_name])

    print("Top parameters sorted by " + metric_name + ":")
    lines_to_file.append("Top parameters sorted by " + metric_name + ":\n")
    for i in range(1, n_params + 1):
        max_value = max(values)
        index = values.index(max_value)
        line = f"{'{0:0=3d}'.format(i)}. Geometric Mean Score:{logs[index]['gmean']} Recall:{logs[index]['recall']} " \
               f"AUC_ROC:{logs[index]['roc_auc']} for params:{logs[index]['params']} " \
               f"with classes_size:{logs[index]['classes_size']}"
        print(line)
        lines_to_file.append(line + '\n')
        values[index] = 0

    return lines_to_file


def logs_to_file(logs: dict, filepath_logs: str, n_params: int):
    with open(filepath_logs + '.txt', 'w') as file:
        top_params_list = (
            top_n_params(logs, 'gmean', n_params), top_n_params(logs, 'recall', n_params),
            top_n_params(logs, 'roc_auc', n_params))

        for lst in top_params_list:
            for elem in lst:
                file.write(elem)
            file.write("----------\n")

        for key in logs.keys():
            file.write(f"{'{0:0=3d}'.format(key)}. {logs[key]}\n")


def draw_plots(parameters_variations: dict, permutations: list, logs: dict, algorithm_name: str, filepath_plots: str):
    gmean_scores = []
    for key in logs.keys():
        gmean_scores.append(logs[key]['gmean'])

    plot_colors = ['blue', 'green', 'orange', 'violet', 'red', 'brown', 'pink']
    fig, axs = plt.subplots(len(parameters_variations.keys()) + 1, **{"figsize": (6.5, 6.5)})
    fig.suptitle(algorithm_name)
    axs[-1].set(xlabel="Experiment Number")

    axs[0].plot(range(1, len(gmean_scores) + 1), gmean_scores, plot_colors[0], markersize=5, marker='o')
    axs[0].set(ylabel="Geometric Mean Score")
    axs[0].grid(True)

    for key in parameters_variations.keys():
        param_values = []
        for exp in permutations:
            param_values.append(exp[key])

        plot_index = list(parameters_variations.keys()).index(key) + 1
        ax = axs[plot_index]
        ax.plot(range(1, len(param_values) + 1), param_values, plot_colors[plot_index], markersize=5, marker='o')
        ax.set(ylabel=key)
        ax.grid(True)

    plt.savefig(filepath_plots + ".png")
    plt.show()


def gridsearch_with_plot(resampler_obj, parameters_dist: dict, X, y, filepath_logs_and_plots: str, n_params: int = 10):
    # Generate permutations of parameters
    keys, values = zip(*parameters_dist.items())
    parameter_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Estimate parameter ranges
    parameter_ranges = {}
    for key in parameters_dist.keys():
        parameter_ranges.update({key: [min(parameters_dist[key]), max(parameters_dist[key])]})
    print("Parameter ranges:", parameter_ranges)
    print("Iterations:", len(parameter_permutations), "\n")

    algorithm_name = resampler_obj.__str__().title().split("(")[0]
    filepath_logs_and_plots += algorithm_name + str(parameter_ranges)

    X = X.values
    y = y.values.ravel()

    logs = {}
    for variant in parameter_permutations:
        resampler_obj.set_params(**variant)
        X_resampled, y_resampled = resampler_obj.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
        clf = RandomForestClassifier(random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if 'estimator' in variant.keys():
            variant["estimator"] = variant['estimator'].__class__.__name__

        index = parameter_permutations.index(variant)
        logs.update({index: {"params": variant, "gmean": geometric_mean_score(y_test, y_pred),
                             "recall": recall_score(y_test, y_pred),
                             "classes_size": count_classes_size(y_resampled),
                             "roc_auc": roc_auc_score(y_test, y_pred)}})

        print('{0:0=3d}'.format(index + 1), logs[index])

    # Drawing plots
    draw_plots(parameters_dist, parameter_permutations, logs, algorithm_name, filepath_logs_and_plots)

    # Print and save top found parameters
    logs_to_file(logs, filepath_logs_and_plots, n_params)


if __name__ == "__main__":
    filepath_logs = "../../plots/"
    filepath_data = '../../data/feature_selected_data/imbalance_set.csv'

    data = data_controller.get_categorized_criteo(filepath_data)
    X, y = split_data_on_x_y(data)

    class_size = count_classes_size(y)
    print("Classes size:", class_size)

    max_n_neighbors = class_size[1] - 1

    value_dict = {
        "shuffle": [True, False],
    }

    obj = GlobalCS()
    gridsearch_with_plot(obj, value_dict, X, y, filepath_logs)
