from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, OneSidedSelection, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold


def random_under_sampler_optimized():
    return RandomUnderSampler(sampling_strategy='auto', random_state=0)


def nearmiss_optimized():
    return NearMiss(n_jobs=-1)


def edited_nearest_neighbours_optimized():
    return EditedNearestNeighbours(n_jobs=-1)


def repeated_edited_nearest_neighbours_optimized():
    return RepeatedEditedNearestNeighbours(n_jobs=-1)


def allknn_optimized():
    return AllKNN(n_jobs=-1)


def one_sided_selection_optimized():
    return OneSidedSelection(random_state=0, n_jobs=-1)


def neighbourhood_cleaning_rule_optimized():
    return NeighbourhoodCleaningRule(n_jobs=-1)


def instance_hardness_threshold_optimized():
    return InstanceHardnessThreshold(random_state=0, n_jobs=-1)
