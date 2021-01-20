from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold
from sklearn.linear_model import LogisticRegression


def cluster_centroids_optimized():
    return ClusterCentroids(random_state=0, n_jobs=-1)


def condensed_nearest_neighbours_optimized():
    return CondensedNearestNeighbour(random_state=0, n_jobs=-1)


def edited_nearest_neighbours_optimized():
    return EditedNearestNeighbours(kind_sel='all', n_jobs=-1)


def repeated_edited_nearest_neighbours_optimized():
    return RepeatedEditedNearestNeighbours(n_jobs=-1)


def allknn_optimized():
    return AllKNN(n_jobs=-1)


def instance_hardness_threshold_optimized():
    return InstanceHardnessThreshold(random_state=0,
                                     estimator=LogisticRegression(
                                         solver='lbfgs', multi_class='auto', n_jobs=-1), n_jobs=-1)


def near_miss_optimized():
    return NearMiss(version=1, n_jobs=-1)


def neighbourhood_cleaning_rule_optimized():
    return NeighbourhoodCleaningRule(n_jobs=-1)


def one_sided_selection_optimized():
    return OneSidedSelection(random_state=0, n_jobs=-1)


def random_under_sampler_optimized():
    return RandomUnderSampler(random_state=0)


def tomek_links_optimized():
    return TomekLinks(sampling_strategy='auto', n_jobs=-1)


def balance_all_undersampling(X):
    print("Undersampling methods balancing:")
    # cluster_centroids_optimized(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # condensed_nearest_neighbours_optimized(X, y)
    # edited_nearest_neighbours_optimized(X, y)
    # repeated_edited_nearest_neighbours_optimized(X, y)
    # allknn_optimized(X, y)
    # instance_hardness_threshold_optimized(X, y)
    # near_miss_optimized(X, y)
    # neighbourhood_cleaning_rule_optimized(X, y)
    # one_sided_selection_optimized(X, y)
    random_under_sampler_optimized()
    # tomek_links_optimized(X, y)
