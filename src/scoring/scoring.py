import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


class ScoringAlgs:
    def __init__(self, data_dict) -> None:
        # """keys: algorithm name, values: Y_test values for algorithm"""
        self.correct_labels = {}

        self.predicted_labels = {}

        """values: probabilities for algorithm"""
        # (likely to be just probas for 1, i.e [:,1] of predict_probas)
        self.probas = {}

        """ {algorithm name: roc_auc_scores} """
        # sklearn.metrics.roc_auc_scores (takes correct_labels[x], probas[x][:,1]
        self.auc_scores = {}

        # fprs & tprs calculated with sklearn.metrics.roc_curve(correct_labels[x],probas[x][:,1])
        self.roc_curves = []

        # confusion matrices for plotting and calculating parameters
        self.conf_matrices = {}

        self.model_names = []

        self.init_from_dict(data_dict)
        super().__init__()


    def plot_conf_matrices(self):
        # """Plots confusion matrix from values provided from Piotrek's module"""

        # prepare annotations
        block_labels = ['True Pos', 'False Pos', 'True Neg', 'False Neg']
        conf_matrices_annots = {}
        for key in self.model_names:
            percent_values = ["{:.2%}".format(x / np.sum(self.conf_matrices[key])) for x in self.conf_matrices[key]]
            perc_and_num = [f"{value0}\n\n{value1}\n\n{value2}" for value0, value1, value2 in
                            zip(block_labels, percent_values, self.conf_matrices[key])]

            conf_matrices_annots[key] = np.asarray(perc_and_num).reshape(2, 2)

        # reshape conf_matrices
        for key in self.model_names:
            buffer = self.conf_matrices[key]
            self.conf_matrices[key] = [buffer[:2], buffer[2:4]]

        # prepare plot info
        conf_matrices_length = len(self.model_names)
        subplots_rows = math.ceil(conf_matrices_length / 2)
        subplots_cols = 2 if conf_matrices_length > 1 else 1
        figure_width = 7 if conf_matrices_length < 2 else 14
        figure_height = 6 * math.ceil(conf_matrices_length / 2)

        fig, axes = plt.subplots(subplots_rows, subplots_cols, figsize=(figure_width, figure_height))

        # fill with Nones if model_names_copy is not equal to axes length
        model_names_copy = list(self.conf_matrices.keys())
        if len(model_names_copy) < axes.size:
            model_names_copy.extend([None for _ in range(axes.size - len(model_names_copy))])

        model_names_copy = np.asarray(model_names_copy).reshape(axes.shape)

        # plot graph
        for value in np.nditer(model_names_copy, flags=['refs_ok']):
            row, col = np.argwhere(model_names_copy == value)[0]
            axes[row][col].set_title('Truth table for {}'.format(value))
            if value == None:
                axes[row][col].set_visible(False)
            else:
                sns.heatmap(data=self.conf_matrices[str(value)], annot=conf_matrices_annots[str(value)], fmt='',
                            ax=axes[row][col], cmap='Blues', xticklabels=False, yticklabels=False)
                plt.setp(axes[row][col], xlabel='Predicted labels', ylabel='True labels')
        plt.savefig('aaaaaa.jpeg')
        plt.close()

    def plot_roc_auc(self):
        # prepare data
        line_styles = ['-', '--', '-.', ':']
        model_roc_curves = []
        for model_name in self.model_names:
            self.auc_scores[model_name] = roc_auc_score(self.correct_labels[model_name], self.probas[model_name])
            model_fpr, model_tpr, _ = roc_curve(self.correct_labels[model_name], self.probas[model_name])
            model_roc_curves.append({'model': model_name, 'fpr': model_fpr, 'tpr': model_tpr})

        # plot
        for x in model_roc_curves:
            plt.plot(x['fpr'], x['tpr'], label=x['model'], linestyle=line_styles[model_roc_curves.index(x)])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.savefig('bbbb.jpeg')
        plt.close()

    def init_from_dict(self, dict_data, add_ns_probs=False):
        results, preds, real = self.unpack_models_dict(dict_input=dict_data)
        self.model_names = list(map(lambda x: x['model'], results))
        truth_table_keys = ['TP', 'FN', 'TN', 'FP']
        for x in self.model_names:
            self.correct_labels[x] = real[self.model_names.index(x)]
            self.predicted_labels[x] = preds[self.model_names.index(x)]
            self.probas[x] = results[self.model_names.index(x)]['predict_proba'][:, 1]
            self.conf_matrices[x] = [results[self.model_names.index(x)][key] for key in truth_table_keys]
        # if add_ns_probs:
        #     self.correct_labels['no_skill'] = real[0]
        #     self.predicted_labels['no_skill'] = [0 for _ in range(len(preds[0]))]

    def unpack_models_dict(self, dict_input):
        results = dict_input['results']
        preds = dict_input['pred']
        real = dict_input['real']
        return results, preds, real
