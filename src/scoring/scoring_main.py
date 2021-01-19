from src.scoring.scoring import ScoringAlgs
import pickle


# przykład użycia ScoringAlgs

def main():
    # testowa funkcja do rozpakowania danych z pickla, normalnie do scoring_algs wchodzi to, co Piotr mi daje
    results = flatten_data_dict(pickle.load(open('wyniki.pickle', 'rb')))

    balanced_results = parseDict(results, test_type='balanced')
    scoring_algs = ScoringAlgs(balanced_results)
    scoring_algs.plot_conf_matrices('balanced_conf')
    scoring_algs.plot_roc_auc('unbalanced_roc')
    other_scores = scoring_algs.calculate_other_measures()
    print(other_scores)

    unbalanced_results = parseDict(results, test_type='unbalanced')
    scoring_algs = ScoringAlgs(unbalanced_results)
    scoring_algs.plot_conf_matrices('unbalanced_conf')
    scoring_algs.plot_roc_auc('unbalanced_roc')
    other_scores = scoring_algs.calculate_other_measures()
    print(other_scores)

    pass


def flatten_data_dict(datadict):
    for x in datadict.keys():
        datadict[x] = datadict[x][0]
    return datadict


def parseDict(data_dict, test_type = 'balanced'):
    result = {}
    for x in data_dict.keys():
        result[x] = {}
        result[x]['train_time'] = data_dict[x]['train_time']
        result[x]['results'] = data_dict[x][test_type]
    return result

if __name__ == "__main__":
    main()
