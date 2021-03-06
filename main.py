import click
import time

from src.balancing.resampler import Resampler
from src.model_builder.model_builder import ModelBuilder
from src.scoring.scoring_wrapper import scoring_starter


def do_preprocessing():
    pass


DEFAULT_SEPARATOR = ','
DEFAULT_YLABEL = 'Sales'

ALL_MODELS = ['xgb', 'cat', 'reg', 'forest']
ALL_BALANCE = ['none', 'ros', 'smotenc', 'rus', 'nearmiss', 'enn', 'renn', 'allknn', 'onesided',
               'ncr', 'iht', 'globalcs', 'soup']


@click.command()
@click.option('-m', '--model', 'model', required=True, multiple=True,
              type=click.Choice(['xgb', 'cat', 'reg', 'tree', 'forest', 'all']), help='Models to train')
@click.option('-b', '--balancing', 'balancing', required=True, multiple=True,
              type=click.Choice(
                  ['none', 'ros', 'smotenc', 'rus', 'nearmiss', 'enn', 'renn', 'allknn', 'onesided',
                   'ncr', 'iht', 'globalcs', 'soup', 'all']), help='Balancing method')
@click.option('-i', '--in', 'in_file', required=False, multiple=False, help='Dataset file')
@click.option('-bd', '--balanced-directory', 'balanced_directory', required=True, multiple=False,
              help='Path to directory containing balanced datasets')
@click.option('-uf', '--unbalanced-filepath', 'unbalanced_filepath', required=True, multiple=False,
              help='Path to unbalanced test dataset', default='data/criteo/criteo_40k.csv')
@click.option('-o', '--out', 'result_directory', required=False, multiple=False,
              help='Directory to save logs and results', default='results_' + str(int(time.time())))
def main(model, balancing, in_file, balanced_directory, result_directory, unbalanced_filepath):
    """
    Bachelor Thesis project.\n
    SRUDA - System for Rating Unbalanced Data Algorithms.\n
    Example run:\n
    ./balancing_main.py -m xgb -i data.csv\n
    ./balancing_main.py -m xgb -m tree -i data2.csv\n
    ./balancing_main.py -m reg -i data.csv -o some_results.csv\n
    """
    if 'all' in balancing:
        balancing = ALL_BALANCE

    if 'all' in model:
        model = ALL_MODELS

    print(model, balancing, in_file, balanced_directory, result_directory, unbalanced_filepath)

    resultDict = {}

    for resampler_name in balancing:

        if resampler_name == 'none':
            balanced_filepath = in_file
        else:
            resampler = Resampler(resampler_name, in_file)
            balanced_filepath = resampler.resample_and_write_to_csv(balanced_directory)

        balancing_method_dict = {}

        for model_name in model:
            builder = ModelBuilder(model_name=model_name,
                                   filename=balanced_filepath,
                                   unbalanced_filename=unbalanced_filepath,
                                   separator=DEFAULT_SEPARATOR,
                                   labels_header=DEFAULT_YLABEL
                                   )
            results, pred_balanced, real_balanced, pred_unbalanced, real_unbalanced = builder.get_result()
            print(results)
            balancing_method_dict[model_name] = results

        resultDict[resampler_name] = balancing_method_dict

    scoring_starter(result_dict=resultDict, base_output_directory="results/")

    return 0


if __name__ == "__main__":
    main()
