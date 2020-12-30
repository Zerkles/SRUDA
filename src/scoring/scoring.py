class ScoringAlgs:
    def __init__(self) -> None:
        """keys: algorithm name, values: Y_test values for algorithm"""
        correct_labels = {}

        """values: probabilities for algorithm"""
        # (likely to be just probas for 1, i.e [:,1] of predict_probas)
        probas = {}

        """ {algorithm name: roc_auc_scores} """
        # sklearn.metrics.roc_auc_scores (takes correct_labels[x], probas[x][:,1]
        auc_scores = {}

        # fprs & tprs calculated with sklearn.metrics.roc_curve(correct_labels[x],probas[x][:,1])
        roc_curves = {}

        # confusion matrices for plotting and calculating parameters
        conf_matrices = {}

        super().__init__()

    """Plots confusion matrix from values provided from Piotrek's module"""
    def plot_conf_matrix(self):
        pass

    """Plots roc from roc_curves values"""
    def plot_roc_auc(self):
        pass

    # add roc_auc_data to existing ScoringAlgs roc list
    """concatenate new roc_auc score"""
    def add_roc_auc_data(self, test_y, probs):
        pass
