from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, OneSidedSelection, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold
from sklearn.ensemble import GradientBoostingClassifier


def random_under_sampler_optimized():
    return RandomUnderSampler(sampling_strategy='auto', random_state=0, replacement=False)


def nearmiss_optimized():
    return NearMiss(sampling_strategy='auto', version=3, n_neighbors=327, n_neighbors_ver3=176, n_jobs=-1)


def edited_nearest_neighbours_optimized():
    return EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=26, kind_sel='all', n_jobs=-1)


def repeated_edited_nearest_neighbours_optimized():
    return RepeatedEditedNearestNeighbours(sampling_strategy='auto', n_neighbors=14, max_iter=3, kind_sel='all',
                                           n_jobs=-1)


def allknn_optimized():
    return AllKNN(sampling_strategy='auto', n_neighbors=66, kind_sel='all', allow_minority=True, n_jobs=-1)


def one_sided_selection_optimized():
    return OneSidedSelection(sampling_strategy='auto', random_state=0, n_neighbors=78, n_seeds_S=21312, n_jobs=-1)


def neighbourhood_cleaning_rule_optimized():
    return NeighbourhoodCleaningRule(sampling_strategy='auto', n_neighbors=32, kind_sel='all', threshold_cleaning=0.5,
                                     n_jobs=-1)


def instance_hardness_threshold_optimized():
    return InstanceHardnessThreshold(estimator=GradientBoostingClassifier(), sampling_strategy='auto', random_state=0,
                                     cv=6, n_jobs=-1)
