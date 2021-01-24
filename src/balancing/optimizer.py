import itertools
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.balancing.data_controller import DataController
from src.balancing.resampler import Resampler


class Optimizer:

    def __init__(self, resampler: Resampler, filepath_for_logs_and_plots: str):
        self.resampler = resampler
        self.filepath_for_logs_and_plots = filepath_for_logs_and_plots

        self.parameters_dist = {}
        self.parameters_perm = []
        self.logs = {}

    def __top_n_params(self, metric_name: str, n_params: int) -> list:
        values = []
        lines_to_file = []
        for key in self.logs.keys():
            values.append(self.logs[key][metric_name])

        print("Top parameters sorted by " + metric_name + ":")
        lines_to_file.append("Top parameters sorted by " + metric_name + ":\n")
        for i in range(1, n_params + 1):
            max_value = max(values)
            index = values.index(max_value)
            line = f"{'{0:0=3d}'.format(i)}. Geometric Mean Score:{self.logs[index]['gmean']} Recall:{self.logs[index]['recall']} " \
                   f"AUC_ROC:{self.logs[index]['roc_auc']} for params:{self.logs[index]['params']} " \
                   f"with classes_size:{self.logs[index]['classes_size']}"
            print(line)
            lines_to_file.append(line + '\n')
            values[index] = 0

        return lines_to_file

    def __logs_to_file(self, filename: str, n_params: int):
        with open(filepath_logs + filename + '.txt', 'w') as file:
            top_params_list = (
                self.__top_n_params('gmean', n_params), self.__top_n_params('recall', n_params),
                self.__top_n_params('roc_auc', n_params))

            for lst in top_params_list:
                for elem in lst:
                    file.write(elem)
                file.write("----------\n")

            for key in self.logs.keys():
                file.write(f"{'{0:0=3d}'.format(key)}. {self.logs[key]}\n")

    def __draw_plots(self, filename: str):
        gmean_scores = []
        for key in self.logs.keys():
            gmean_scores.append(self.logs[key]['gmean'])

        plot_colors = ['blue', 'green', 'orange', 'violet', 'red', 'brown', 'pink']
        fig, axs = plt.subplots(len(self.parameters_dist.keys()) + 1, **{"figsize": (6.5, 6.5)})
        fig.suptitle(self.resampler.get_name())
        axs[-1].set(xlabel="Experiment Number")

        axs[0].plot(range(1, len(gmean_scores) + 1), gmean_scores, plot_colors[0], markersize=5, marker='o')
        axs[0].set(ylabel="Geometric Mean Score")
        axs[0].grid(True)

        for key in self.parameters_dist.keys():
            param_values = []
            for exp in self.parameters_perm:
                param_values.append(exp[key])

            plot_index = list(self.parameters_dist.keys()).index(key) + 1
            ax = axs[plot_index]
            ax.plot(range(1, len(param_values) + 1), param_values, plot_colors[plot_index], markersize=5, marker='o')
            ax.set(ylabel=key)
            ax.grid(True)

        plt.savefig(self.filepath_for_logs_and_plots + filename + ".png")
        plt.show()

    def __generate_permutations(self):
        # Generate permutations of parameters
        keys, values = zip(*self.parameters_dist.items())
        self.parameters_perm = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def __display_research_info(self) -> dict:
        # Estimate parameter ranges
        parameter_ranges = {}
        for key in self.parameters_dist.keys():
            parameter_ranges.update({key: f"{self.parameters_dist[key][0]}-{self.parameters_dist[key][-1]}"})
        print("Parameter ranges:", parameter_ranges)
        print("Iterations:", len(self.parameters_perm), "\n")
        return parameter_ranges

    def run_research(self, parameters_dist: dict, n_params: int = 10):
        self.parameters_dist = parameters_dist
        self.__generate_permutations()

        parameter_ranges = self.__display_research_info()

        for variant in self.parameters_perm:
            self.resampler.set_params(**variant)
            X_resampled, y_resampled = self.resampler.resample_to_ndarray()

            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
            clf = RandomForestClassifier(random_state=0, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            if 'estimator' in variant.keys():
                if variant['estimator'].__class__.__name__ == 'RandomForestClassifier':
                    variant['estimator'] = 'RFC'
                elif variant['estimator'].__class__.__name__ == 'AdaBoostClassifier':
                    variant['estimator'] = 'ABC'
                elif variant['estimator'].__class__.__name__ == 'GradientBoostingClassifier':
                    variant['estimator'] = 'GBC'

            index = self.parameters_perm.index(variant)
            self.logs.update({index: {"params": variant, "gmean": geometric_mean_score(y_test, y_pred),
                                      "recall": recall_score(y_test, y_pred),
                                      "classes_size": DataController.count_classes_size(y_resampled),
                                      "roc_auc": roc_auc_score(y_test, y_pred)}})

            print('{0:0=3d}'.format(index + 1), self.logs[index])

        filename = self.resampler.get_name() + str(parameter_ranges)

        # Drawing plots
        self.__draw_plots(filename)

        # Print and save top found parameters
        self.__logs_to_file(filename, n_params)


def get_n_elements_from_list(original_list: list, n_elements: int):
    if n_elements >= len(original_list):
        return original_list
    elif type(len(original_list) / n_elements) != int:
        n_elements -= 1

    step = int(len(original_list) / n_elements)
    if step == 0:
        step = 1

    return original_list[::step]


if __name__ == "__main__":
    filepath_logs = "../../plots/"
    filepath_data = '../../data/feature_selected_data/imbalance_set.csv'

    res = Resampler('iht', filepath_data)
    res.set_params(**{'sampling_strategy': 'auto', 'random_state': 0, 'n_jobs': -1})  # setting common params

    max_n_neighbors = DataController.count_classes_size(res.y_unbalanced)[1] - 1

    value_dict = {
        "estimator": [AdaBoostClassifier(random_state=0),
                      GradientBoostingClassifier(random_state=0)],
        "cv": get_n_elements_from_list(list(range(2, 8)), 11)
    }

    opt = Optimizer(res, filepath_logs)
    opt.run_research(value_dict)
