from src.scoring.scoring import ScoringAlgs


def main():
    import pickle
    scoringAlgs = ScoringAlgs(pickle.load(open('../../cos.pickle', 'rb')))
    scoringAlgs.plot_conf_matrices()
    scoringAlgs.plot_roc_auc()
    pass


if __name__ == "__main__":
    main()
