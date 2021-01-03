from src.scoring.scoring import ScoringAlgs
import pickle

# przykład użycia ScoringAlgs

def main():
    # testowa funkcja do rozpakowania danych z pickla, normalnie do scoring_algs wchodzi to, co Piotr mi daje
    results, preds, real = unpack_models_dict(pickle.load(open('wyniki.pickle', 'rb')))

    scoring_algs = ScoringAlgs(results, preds, real)
    scoring_algs.plot_conf_matrices()
    scoring_algs.plot_roc_auc()
    pass


def unpack_models_dict(dict_input):
    results = dict_input['results']
    preds = dict_input['pred']
    real = dict_input['real']
    return results, preds, real


if __name__ == "__main__":
    main()
