import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import make_classification

from balancing.over_sampling import train_and_score
from data import data_controller


def cluster_centroids(X, y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def random_under_sampler(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled

    print("   with replacement:")
    import numpy as np
    print(np.vstack([tuple(row) for row in X_resampled]).shape)

    rus = RandomUnderSampler(random_state=0, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(np.vstack(np.unique([tuple(row) for row in X_resampled], axis=0)).shape)
    return X_resampled, y_resampled


def near_miss(X, y, version: int):
    from imblearn.under_sampling import NearMiss
    nm = NearMiss(version=version)
    X_resampled, y_resampled = nm.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def tomek_links(X, y, sampling_strategy: str):
    from imblearn.under_sampling import TomekLinks
    tl = TomekLinks(sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = tl.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def edited_nearest_neighbours(X, y, kind_sel: str):
    sorted(Counter(y).items())

    from imblearn.under_sampling import EditedNearestNeighbours
    enn = EditedNearestNeighbours(kind_sel=kind_sel)
    X_resampled, y_resampled = enn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def repeated_edited_nearest_neighbours(X, y):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    renn = RepeatedEditedNearestNeighbours()
    X_resampled, y_resampled = renn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def allknn(X, y):
    from imblearn.under_sampling import AllKNN
    allknn = AllKNN()
    X_resampled, y_resampled = allknn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def condensed_nearest_neighbours(X, y):
    from imblearn.under_sampling import CondensedNearestNeighbour
    cnn = CondensedNearestNeighbour(random_state=0)
    X_resampled, y_resampled = cnn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def one_sided_selection(X, y):
    from imblearn.under_sampling import OneSidedSelection
    oss = OneSidedSelection(random_state=0)
    X_resampled, y_resampled = oss.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def neighbourhood_cleaning_rule(X, y):
    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule()
    X_resampled, y_resampled = ncr.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def instance_hardness_threshold(X, y):
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import InstanceHardnessThreshold
    iht = InstanceHardnessThreshold(random_state=0,
                                    estimator=LogisticRegression(
                                        solver='lbfgs', multi_class='auto'))
    X_resampled, y_resampled = iht.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def compare_balancing_methods(X, y):
    print("ClusterClassification:")
    X_resampled, y_resampled = cluster_centroids(X, y)
    train_and_score(X_resampled, y_resampled)

    print("RandomUnderSampling:")
    X_resampled, y_resampled = random_under_sampler(X, y)
    train_and_score(X_resampled, y_resampled)

    print("NearMiss:")
    X_resampled, y_resampled = near_miss(X, y, version=1)
    train_and_score(X_resampled, y_resampled)

    print('TomekLinks:')
    X_resampled, y_resampled = tomek_links(X, y, sampling_strategy='auto')
    train_and_score(X_resampled, y_resampled)

    print('Edited Nearest Neighbours:')
    X_resampled, y_resampled = edited_nearest_neighbours(X, y, kind_sel='all')
    train_and_score(X_resampled, y_resampled)
    print('Repeated Edited Nearest Neighbours:')
    X_resampled, y_resampled = repeated_edited_nearest_neighbours(X, y)
    train_and_score(X_resampled, y_resampled)
    print('AllKNN:')
    X_resampled, y_resampled = allknn(X, y)
    train_and_score(X_resampled, y_resampled)

    print("Condensed Nearest Neighbours:")
    X_resampled, y_resampled = condensed_nearest_neighbours(X, y)
    train_and_score(X_resampled, y_resampled)

    print("One Sided Selection:")
    X_resampled, y_resampled = one_sided_selection(X, y)
    train_and_score(X_resampled, y_resampled)

    print("Neighbourhood Cleaning Rule")
    X_resampled, y_resampled = neighbourhood_cleaning_rule(X, y)
    train_and_score(X_resampled, y_resampled)

    # print("Instance Hardness Threshold")
    # X_resampled, y_resampled = instance_hardness_threshold(X, y)
    # train_and_score(X_resampled, y_resampled)
    # TODO: tu coś się wywala


if __name__ == '__main__':
    samples_count = 10000
    X_original, y_original = data_controller.get_converted_data(samples_count)
    print(sorted(Counter(y_original).items()))

    print("Score for not balanced data:")
    train_and_score(X_original, y_original)

    compare_balancing_methods(X_original, y_original)
