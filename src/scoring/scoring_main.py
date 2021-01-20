import os

from tqdm import tqdm

from src.scoring.scoring import ScoringAlgs
import pickle


def scoring_starter(result_dict, base_output_directory):
    balancing_methods_keys = result_dict.keys()
    scores = {}
    for balancing_method in tqdm(balancing_methods_keys):
        scores[balancing_method] = {}
        output_directory = base_output_directory + balancing_method
        make_dir_if_not_present(output_directory)

        scores[balancing_method]['balanced'] = execute_scoring_routine(data=result_dict,
                                                                       balancing_method=balancing_method,
                                                                       output_dir=output_directory,
                                                                       test_type='balanced')
        scores[balancing_method]['unbalanced'] = execute_scoring_routine(data=result_dict,
                                                                         balancing_method=balancing_method,
                                                                         output_dir=output_directory,
                                                                         test_type='unbalanced',
                                                                         scoring_measures=['auroc'])

    # todo export scores to dataframes (nie pamiętam jak te tabele miały wyglądać)
    import pickle
    with open('scores.pickle', 'wb') as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)


def execute_scoring_routine(data=None, balancing_method="", output_dir="", test_type='balanced', scoring_measures=None):
    """
    Saves plots and scores of trained model
    @param data: data dictionary
    @param balancing_method: key of balancing method used before training a model
    @param output_dir: output directory for plotted scores
    @param test_type: balanced/imbalanced
    @param scoring_measures: list of scoring measures that will be saved/plotted;
    one or more options: "balanced_acc_score", "acc", "recall", "f1" or "auroc";
    providing no values defaults to using all of them
    @return: None
    """

    results = extract_selected_test_type(data[balancing_method], test_type=test_type)
    scoring_algs = ScoringAlgs(results)
    scoring_algs.plot_conf_matrices('{}/{}_balanced_conf'.format(output_dir, balancing_method))
    scoring_algs.plot_roc_auc('{}/{}_balanced_roc'.format(output_dir, balancing_method))
    other_scores = scoring_algs.calculate_other_measures(scores=scoring_measures)
    # todo export scores to csv
    # print(other_scores)
    return other_scores


def main():
    results = pickle.load(open('rezultat.pickle', 'rb'))
    scoring_starter(results, '')


def extract_selected_test_type(data_dict, test_type='balanced'):
    result = {}
    for x in data_dict.keys():
        result[x] = {}
        result[x]['train_time'] = data_dict[x]['train_time']
        result[x]['results'] = data_dict[x][test_type]
    return result


def make_dir_if_not_present(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""
    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


if __name__ == "__main__":
    main()
