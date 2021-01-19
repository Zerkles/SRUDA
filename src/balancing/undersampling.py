from cmath import sqrt
from collections import Counter

from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule

from src.balancing.utilities import resample_and_write_to_csv, feature_graph_generator


def cluster_centroids_variations(X, y):
    obj = ClusterCentroids(random_state=0, n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "ClusterCentroids" + str(obj.get_params()))


def condensed_nearest_neighbours_variations(X, y):
    obj = CondensedNearestNeighbour(random_state=0, n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "CondensedNearestNeighbours" + str(obj.get_params()))


def edited_nearest_neighbours_variations(X, y):
    obj = EditedNearestNeighbours(kind_sel='all', n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "EditedNearestNeighbours" + str(obj.get_params()))


def repeated_edited_nearest_neighbours_variations(X, y):
    obj = RepeatedEditedNearestNeighbours(n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "RepeatedEditedNearestNeighbours" + str(obj.get_params()))


def allknn_variations(X, y):
    obj = AllKNN(n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "AllKNN" + str(obj.get_params()))


def instance_hardness_threshold_variations(X, y):
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import InstanceHardnessThreshold
    obj = InstanceHardnessThreshold(random_state=0,
                                    estimator=LogisticRegression(
                                        solver='lbfgs', multi_class='auto', n_jobs=-1), n_jobs=-1)

    # resample_and_write_to_csv(obj, X, y, "InstanceHardnessThreshold" + str(obj.get_params()))
    resample_and_write_to_csv(obj, X, y, "InstanceHardnessThreshold")


def near_miss_variations(X, y):
    obj = NearMiss(version=1, n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "NearMiss" + str(obj.get_params()))


def neighbourhood_cleaning_rule_variations(X, y):
    obj = NeighbourhoodCleaningRule(n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "NeighbourhoodCleaningRule" + str(obj.get_params()))


def one_sided_selection_variations(X, y):
    obj = OneSidedSelection(random_state=0, n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "OneSidedSelection" + str(obj.get_params()))


def random_under_sampler_variations(X, y):
    obj = RandomUnderSampler(random_state=0)
    resample_and_write_to_csv(obj, X, y, "RandomUnderSampler" + str(obj.get_params()))


def tomek_links_variations(X, y):
    obj = TomekLinks(sampling_strategy='auto', n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "TomekLinks" + str(obj.get_params()))


def balance_all_undersampling(X, y):
    print("Undersampling methods balancing:")
    # cluster_centroids_variations(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # condensed_nearest_neighbours_variations(X, y)
    # edited_nearest_neighbours_variations(X, y)
    # repeated_edited_nearest_neighbours_variations(X, y)
    # allknn_variations(X, y)
    # instance_hardness_threshold_variations(X, y)
    # near_miss_variations(X, y)
    # neighbourhood_cleaning_rule_variations(X, y)
    # one_sided_selection_variations(X, y)
    random_under_sampler_variations(X, y)
    # tomek_links_variations(X, y)
