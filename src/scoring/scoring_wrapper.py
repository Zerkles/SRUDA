import os

from tqdm import tqdm
import pandas as pd
from src.scoring.scoring import ScoringAlgs


def scoring_starter(result_dict, base_output_directory):
    scores_dict = run_scoring(result_dict, base_output_directory)
    balanced_results_dict, unbalanced_results_dict = parse_to_dict_of_dfs(scores_dict)
    export_results_to_csvs(balanced_results_dict, base_output_directory, score_type='balanced')
    export_results_to_csvs(unbalanced_results_dict, base_output_directory, score_type='unbalanced')


def export_results_to_csvs(dict_of_dataframes, base_output_directory, score_type=''):
    for score_name in dict_of_dataframes.keys():
        dict_of_dataframes[score_name].to_csv("{}/{}_{}.csv".format(base_output_directory, score_type, score_name))


def parse_to_dict_of_dfs(scores_dict):
    return parse_scores_dict(scores_dict, balancing_type='balanced'), parse_scores_dict(scores_dict,
                                                                                        balancing_type='unbalanced')


def run_scoring(result_dict, base_output_directory):
    balancing_methods_keys = result_dict.keys()
    scores = {}
    pbar = tqdm(balancing_methods_keys)
    for balancing_method in pbar:
        pbar.set_description("Scoring %s results" % balancing_method)
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
                                                                         scoring_measures=['auroc', 'test_time'])

    return scores


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
    scoring_algs.plot_conf_matrices('{}/{}_{}_conf'.format(output_dir, balancing_method, test_type))
    scoring_algs.plot_roc_auc('{}/{}_{}_roc'.format(output_dir, balancing_method, test_type))
    other_scores = scoring_algs.calculate_other_measures(scores=scoring_measures)
    return other_scores


def extract_selected_test_type(data_dict, test_type='balanced'):
    result = {}
    for x in data_dict.keys():
        result[x] = {}
        result[x]['train_time'] = data_dict[x]['train_time']
        result[x]['results'] = data_dict[x][test_type]
        result[x]['test_time'] = data_dict[x][test_type]['test_time']
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


def parse_scores_dict(scores_file, balancing_type='balanced'):
    cols_data = {}
    column_labels = {}
    tables_cols = {}
    dict_of_results = {}

    for x in scores_file.keys():
        for y in scores_file[x][balancing_type].keys():
            if y in tables_cols.keys():
                tables_cols[y][x] = scores_file[x][balancing_type][y]
            else:
                tables_cols[y] = {}
                tables_cols[y][x] = scores_file[x][balancing_type][y]

    for x in tables_cols.keys():
        for y in tables_cols[x].keys():
            if x not in column_labels.keys():
                column_labels[x] = []
            column_labels[x].append(y)
            for z in tables_cols[x][y].keys():
                if x not in cols_data.keys():
                    cols_data[x] = {}
                if z not in cols_data[x].keys():
                    cols_data[x][z] = []
                cols_data[x][z].append(tables_cols[x][y][z])

    for x in cols_data.keys():
        df = pd.DataFrame.from_dict(cols_data[x], orient='index', columns=column_labels[x])
        dict_of_results[x] = df

    #     don't ask me how it works, but it does
    return dict_of_results
