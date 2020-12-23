from collections import Counter


def cluster_centroids(X, y, cores_count):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0, n_jobs=cores_count)
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


def near_miss(X, y, cores_count, version: int):
    from imblearn.under_sampling import NearMiss
    nm = NearMiss(version=version, n_jobs=cores_count)
    X_resampled, y_resampled = nm.fit_resample(X, y)
    return X_resampled, y_resampled


def tomek_links(X, y, cores_count, sampling_strategy: str):
    from imblearn.under_sampling import TomekLinks
    tl = TomekLinks(sampling_strategy=sampling_strategy, n_jobs=cores_count)
    X_resampled, y_resampled = tl.fit_resample(X, y)
    return X_resampled, y_resampled


def edited_nearest_neighbours(X, y, cores_count, kind_sel: str):
    sorted(Counter(y).items())

    from imblearn.under_sampling import EditedNearestNeighbours
    enn = EditedNearestNeighbours(kind_sel=kind_sel, n_jobs=cores_count)
    X_resampled, y_resampled = enn.fit_resample(X, y)
    return X_resampled, y_resampled


def repeated_edited_nearest_neighbours(X, y, cores_count):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    renn = RepeatedEditedNearestNeighbours(n_jobs=cores_count)
    X_resampled, y_resampled = renn.fit_resample(X, y)
    return X_resampled, y_resampled


def allknn(X, y, cores_count):
    from imblearn.under_sampling import AllKNN
    allknn = AllKNN(n_jobs=cores_count)
    X_resampled, y_resampled = allknn.fit_resample(X, y)
    return X_resampled, y_resampled


def condensed_nearest_neighbours(X, y, cores_count):
    from imblearn.under_sampling import CondensedNearestNeighbour
    cnn = CondensedNearestNeighbour(random_state=0, n_jobs=cores_count)
    X_resampled, y_resampled = cnn.fit_resample(X, y)
    return X_resampled, y_resampled


def one_sided_selection(X, y, cores_count):
    from imblearn.under_sampling import OneSidedSelection
    oss = OneSidedSelection(random_state=0, n_jobs=cores_count)
    X_resampled, y_resampled = oss.fit_resample(X, y, )
    return X_resampled, y_resampled


def neighbourhood_cleaning_rule(X, y, cores_count):
    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule(n_jobs=cores_count)
    X_resampled, y_resampled = ncr.fit_resample(X, y)
    return X_resampled, y_resampled


def instance_hardness_threshold(X, y, cores_count):
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import InstanceHardnessThreshold
    iht = InstanceHardnessThreshold(random_state=0,
                                    estimator=LogisticRegression(
                                        solver='lbfgs', multi_class='auto'), n_jobs=cores_count)
    X_resampled, y_resampled = iht.fit_resample(X, y)
    return X_resampled, y_resampled
