import click
import time

from src.balancing.utilities import funkcja
from src.model_builder.model_builder import ModelBuilder


def do_preprocessing():
    pass


@click.command()
@click.option('-m', '--model', 'model', required=True, multiple=True,
              type=click.Choice(['xgb', 'cat', 'reg', 'tree'], case_sensitive=False), help='Models to train')
@click.option('-b', '--balancing', 'balancing', required=False, multiple=False,
              type=click.Choice(['ros', 'rus'], case_sensitive=False), help='Balancing techniques applied')
@click.option('-i', '--in', 'in_file', required=False, multiple=False, help='Dataset file')
@click.option('-uf', '--unbalanced-filepath', 'unbalanced_filepath', required=True, multiple=False,
              help='Path to unbalanced test dataset', default='data/criteo/criteo_40k.csv')
@click.option('-o', '--out', 'result_directory', required=False, multiple=False,
              help='Directory to save logs and results', default='results_' + str(int(time.time())))
def main(model, balancing, in_file, result_directory, unbalanced_filepath):
    """
    Bachelor Thesis project.\n
    SRUDA - System for Rating Unbalanced Data Algorithms.\n
    Example run:\n
    ./main.py -m xgb -i data.csv\n
    ./main.py -m xgb -m tree -i data2.csv\n
    ./main.py -m reg -i data.csv -o some_results.csv\n
    """
    print(model, in_file, result_directory)
    # preprocessing
    # balancing
    # model building

    balanced_filepath = funkcja(balancing, in_file)

    for name in model:
        builder = ModelBuilder(model_name=name,
                               filename=balanced_filepath,
                               unbalanced_filename=unbalanced_filepath,
                               separator=',',
                               labels_header='Sales'
                               )
        results, pred_balanced, real_balanced, pred_unbalanced, real_unbalanced = builder.get_result()
        print(results)
    d = {
        "filename": "data_set_1",
        "time": 24.3,
        "TP": 100,
        "TN": 100,
        "FP": 10,
        "FN": 10,
        "score": (0, 2, 4, 4)
    }

    # ---------------------

    return 0


if __name__ == "__main__":
    main()
