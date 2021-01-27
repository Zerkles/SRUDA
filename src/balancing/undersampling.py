from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, OneSidedSelection, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold
from sklearn.ensemble import GradientBoostingClassifier


def random_under_sampler_optimized():
    return RandomUnderSampler()


def nearmiss_optimized():
    return NearMiss()


def tomek_links_optimized():
    return TomekLinks()


def edited_nearest_neighbours_optimized():
    return EditedNearestNeighbours()


def repeated_edited_nearest_neighbours_optimized():
    return RepeatedEditedNearestNeighbours()


def allknn_optimized():
    return AllKNN()


def one_sided_selection_optimized():
    return OneSidedSelection()


def neighbourhood_cleaning_rule_optimized():
    return NeighbourhoodCleaningRule()


def instance_hardness_threshold_optimized():
    return InstanceHardnessThreshold()
