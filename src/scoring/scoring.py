import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, accuracy_score, recall_score, f1_score


class ScoringAlgs:
    def __init__(self, results) -> None:
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

        self.init_from_dict(results)
        super().__init__()

    def plot_conf_matrices(self, filename='confusion_matrices'):
        # """Plots confusion matrix from values provided from Piotrek's module"""

        # prepare annotations
        block_labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
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

        # dopóki w tym result nie mam jeszcze zapisanego sposobu balansowania ani preprocessingu, to zapisuję tylko tutaj
        plt.savefig(filename+'.jpeg')
        plt.close()

    def calculate_roc_auc(self):
        # prepare data
        line_styles = ['-', '--', '-.', ':']
        model_roc_curves = []
        for model_name in self.model_names:
            self.auc_scores[model_name] = roc_auc_score(self.correct_labels[model_name], self.probas[model_name])
            model_fpr, model_tpr, _ = roc_curve(self.correct_labels[model_name], self.probas[model_name])
            model_roc_curves.append({'model': model_name, 'fpr': model_fpr, 'tpr': model_tpr})
        self.roc_curves = model_roc_curves

    def plot_roc_auc(self, filename = 'ROCs'):
        self.calculate_roc_auc()
        line_styles = ['-', '--', '-.', ':']
        for x in self.roc_curves:
            plt.plot(x['fpr'], x['tpr'], label=x['model'], linestyle=line_styles[self.roc_curves.index(x)])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        # dopóki w tym result nie mam jeszcze zapisanego sposobu balansowania ani preprocessingu, to zapisuję tylko tutaj
        plt.savefig(filename+'.jpeg')
        plt.close()

    def set_model_names(self, new_model_names):
        self.model_names = new_model_names

    def init_from_dict(self, result_dict):
        self.set_model_names(result_dict.keys())
        truth_table_keys = ['TN', 'FP', 'FN', 'TP']

        for model_name in self.model_names:
            self.correct_labels[model_name] = result_dict[model_name]['results']['real']
            self.predicted_labels[model_name] = result_dict[model_name]['results']['predicted']
            self.probas[model_name] = result_dict[model_name]['results']['predict_proba'][:, 1]
            self.conf_matrices[model_name] = [result_dict[model_name]['results'][key] for key in truth_table_keys]
        # if add_ns_probs:
        #     self.correct_labels['no_skill'] = real[0]
        #     self.predicted_labels['no_skill'] = [0 for _ in range(len(preds[0]))]

    def calculate_other_measures(self, y_pred=None, y_true=None, balanced_acc_score=True, acc=True, recall=True,
                                 f1=True):
        resultDict = {}
        if balanced_acc_score:
            resultDict['balanced_acc_score'] = self.calculate_score(balanced_accuracy_score)
        if acc:
            resultDict['accuracy'] = self.calculate_score(accuracy_score)
        if recall:
            resultDict['recall'] = self.calculate_score(recall_score)
        if f1:
            resultDict['f1'] = self.calculate_score(f1_score)
        return resultDict

    def calculate_score(self, func):
        models_scores_dict = {}
        for model_name in self.model_names:
            models_scores_dict[model_name] = func(y_true=self.correct_labels[model_name],
                                                  y_pred=self.predicted_labels[model_name])
        return models_scores_dict