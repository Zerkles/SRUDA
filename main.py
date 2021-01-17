import click
import time
from src.model_builder.model_builder import ModelBuilder


def do_preprocessing():
    pass


@click.command()
@click.option('-m', '--model', 'model', required=True, multiple=True,
              type=click.Choice(['xgb', 'cat', 'reg', 'tree'], case_sensitive=False), help='Models to train')
@click.option('-p', '--pre-processing', 'preprocessing_types', required=False,
              multiple=True, help='Pre-processing algorithms to use')
@click.option('-i', '--in', 'in_file', required=False, multiple=False, help='Dataset file')
@click.option('-o', '--out', 'result_directory', required=False, multiple=False,
              help='Directory to save logs and results', default='results_'+str(int(time.time())))
def main(model, preprocessing_types, in_file, result_directory):
    """
    Bachelor Thesis project.\n
    SRUDA - System for Rating Unbalanced Data Algorithms.\n
    Example run:\n
    ./main.py -m xgb -i data.csv\n
    ./main.py -m xgb -m tree -i data2.csv\n
    ./main.py -m reg -i data.csv -o some_results.csv\n
    """
    print(model, preprocessing_types, in_file, result_directory)
    # preprocessing
    # balancing
    # model building

    unbalanced_name = 'data/criteo/criteo_40k.csv'

    resultDict = {}

    for name in model:
        builder = ModelBuilder(model_name=name,
                               filename=in_file,
                               unbalanced_filename=unbalanced_name,
                               separator='\t',
                               labels_header='Sales'
                               )
        results = builder.get_result()
        resultDict[name] = results,
        # print(results)

    import pickle
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(resultDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # scoring goes here
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

