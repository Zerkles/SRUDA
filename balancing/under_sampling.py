from collections import Counter

from train_score_test import train_and_score, percent_change


def cluster_centroids(X, y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)
    return X_resampled, y_resampled


def random_under_sampler(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
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
    return X_resampled, y_resampled


def tomek_links(X, y, sampling_strategy: str):
    from imblearn.under_sampling import TomekLinks
    tl = TomekLinks(sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = tl.fit_resample(X, y)
    return X_resampled, y_resampled


def edited_nearest_neighbours(X, y, kind_sel: str):
    sorted(Counter(y).items())

    from imblearn.under_sampling import EditedNearestNeighbours
    enn = EditedNearestNeighbours(kind_sel=kind_sel)
    X_resampled, y_resampled = enn.fit_resample(X, y)
    return X_resampled, y_resampled


def repeated_edited_nearest_neighbours(X, y):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    renn = RepeatedEditedNearestNeighbours()
    X_resampled, y_resampled = renn.fit_resample(X, y)
    return X_resampled, y_resampled


def allknn(X, y):
    from imblearn.under_sampling import AllKNN
    allknn = AllKNN()
    X_resampled, y_resampled = allknn.fit_resample(X, y)
    return X_resampled, y_resampled


def condensed_nearest_neighbours(X, y):
    from imblearn.under_sampling import CondensedNearestNeighbour
    cnn = CondensedNearestNeighbour(random_state=0)
    X_resampled, y_resampled = cnn.fit_resample(X, y)
    return X_resampled, y_resampled


def one_sided_selection(X, y):
    from imblearn.under_sampling import OneSidedSelection
    oss = OneSidedSelection(random_state=0)
    X_resampled, y_resampled = oss.fit_resample(X, y)
    return X_resampled, y_resampled


def neighbourhood_cleaning_rule(X, y):
    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule()
    X_resampled, y_resampled = ncr.fit_resample(X, y)
    return X_resampled, y_resampled


def instance_hardness_threshold(X, y):
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import InstanceHardnessThreshold
    iht = InstanceHardnessThreshold(random_state=0,
                                    estimator=LogisticRegression(
                                        solver='lbfgs', multi_class='auto'))
    X_resampled, y_resampled = iht.fit_resample(X, y)
    return X_resampled, y_resampled


def compare_undersampling_methods(X, y, score_original):
    print("ClusterCentroids:")
    X_resampled, y_resampled = cluster_centroids(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("RandomUnderSampling:")
    X_resampled, y_resampled = random_under_sampler(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("NearMiss:")
    X_resampled, y_resampled = near_miss(X, y, version=1)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print('TomekLinks:')
    X_resampled, y_resampled = tomek_links(X, y, sampling_strategy='auto')
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print('Edited Nearest Neighbours:')
    X_resampled, y_resampled = edited_nearest_neighbours(X, y, kind_sel='all')
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print('Repeated Edited Nearest Neighbours:')
    X_resampled, y_resampled = repeated_edited_nearest_neighbours(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print('AllKNN:')
    X_resampled, y_resampled = allknn(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("Condensed Nearest Neighbours:")
    X_resampled, y_resampled = condensed_nearest_neighbours(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("One Sided Selection:")
    X_resampled, y_resampled = one_sided_selection(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("Neighbourhood Cleaning Rule")
    X_resampled, y_resampled = neighbourhood_cleaning_rule(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    # print("Instance Hardness Threshold")
    # X_resampled, y_resampled = instance_hardness_threshold(X, y)
    # train_and_score(X_resampled, y_resampled)
    # TODO: tu coś się wywala
