from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule

from src.balancing.utilities import resample_and_write_to_csv


def cluster_centroids_variations(X, y, cores_count):
    obj = ClusterCentroids(random_state=0, n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "ClusterCentroids" + str(obj.get_params()))


def random_under_sampler_variations(X, y):
    obj = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = resample_and_write_to_csv(obj, X, y, "RandomOverSampler" + str(obj.get_params()))

    print("   with replacement:")
    # print(np.vstack([tuple(row) for row in X_resampled]).shape)

    obj = RandomUnderSampler(random_state=0, replacement=True)
    # X_resampled, y_resampled = resample_and_write_to_csv(obj, X, y, "RandomOverSampler" + str(obj.get_params()))
    # print(np.vstack(np.unique([tuple(row) for row in X_resampled], axis=0)).shape)


def near_miss_variations(X, y, cores_count):
    obj = NearMiss(version=1, n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "NearMiss" + str(obj.get_params()))


def tomek_links_variations(X, y, cores_count):
    obj = TomekLinks(sampling_strategy='auto', n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "TomekLinks" + str(obj.get_params()))


def edited_nearest_neighbours_variations(X, y, cores_count):
    obj = EditedNearestNeighbours(kind_sel='all', n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "EditedNearestNeighbours" + str(obj.get_params()))


def repeated_edited_nearest_neighbours_variations(X, y, cores_count):
    obj = RepeatedEditedNearestNeighbours(n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "RepeatedEditedNearestNeighbours" + str(obj.get_params()))


def allknn_variations(X, y, cores_count):
    obj = AllKNN(n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "AllKNN" + str(obj.get_params()))


def condensed_nearest_neighbours_variations(X, y, cores_count):
    obj = CondensedNearestNeighbour(random_state=0, n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "CondensedNearestNeighbours" + str(obj.get_params()))


def one_sided_selection_variations(X, y, cores_count):
    obj = OneSidedSelection(random_state=0, n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "OneSidedSelection" + str(obj.get_params()))


def neighbourhood_cleaning_rule_variations(X, y, cores_count):
    obj = NeighbourhoodCleaningRule(n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "NeighbourhoodCleaningRule" + str(obj.get_params()))


def instance_hardness_threshold(X, y, cores_count):
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import InstanceHardnessThreshold
    iht = InstanceHardnessThreshold(random_state=0,
                                    estimator=LogisticRegression(
                                        solver='lbfgs', multi_class='auto'), n_jobs=cores_count)
    X_resampled, y_resampled = iht.fit_resample(X, y)
    return X_resampled, y_resampled


def balance_all_undersampling(X, y, cores_count):
    print("Undersampling methods balancing:")
    # cluster_centroids_variations(X, y, cores_count)
    random_under_sampler_variations(X, y)
    #near_miss_variations(X, y, cores_count)
    #tomek_links_variations(X, y, cores_count)
    #edited_nearest_neighbours_variations(X, y, cores_count)
    #repeated_edited_nearest_neighbours_variations(X, y, cores_count)
    #allknn_variations(X, y, cores_count)
   # condensed_nearest_neighbours_variations(X, y, cores_count)
    #one_sided_selection_variations(X, y, cores_count)
    #neighbourhood_cleaning_rule_variations(X, y, cores_count)

    # balance_to_csv(X, y, score_original, instance_hardness_threshold, "InstanceHardnessThreshold")
    # TODO: tu coś się wywala
