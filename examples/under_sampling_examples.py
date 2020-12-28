import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import make_classification


def get_classified_data():
    # Dzielenie danych na 3 klastry niezbalansowane
    return make_classification(n_samples=5000, n_features=2, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               class_sep=0.8, random_state=0)


def draw_plot(X, y, name: str):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='black', alpha=0.75)
    plt.title(name)
    plt.show()


def cluster_centroids(X, y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'Cluster Centroids')


def random_under_sampler(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'RandomUnderSampler')

    print("   with replacement:")
    import numpy as np
    print(np.vstack([tuple(row) for row in X_resampled]).shape)

    rus = RandomUnderSampler(random_state=0, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(np.vstack(np.unique([tuple(row) for row in X_resampled], axis=0)).shape)
    draw_plot(X_resampled, y_resampled, 'RandomUnderSampler with replacement')

    print("   resampling heterogeneous data:")
    X_hetero = np.array([['xxx', 1, 1.0], ['yyy', 2, 2.0], ['zzz', 3, 3.0]],
                        dtype=np.object)
    y_hetero = np.array([0, 0, 1])
    X_resampled, y_resampled = rus.fit_resample(X_hetero, y_hetero)
    print(X_resampled)
    print(y_resampled)
    draw_plot(X_resampled, y_resampled, 'RandomUnderSampler with heterogeneous data')
    # TODO: coś tu spierdolone jest w wykresie


def near_miss(X, y, version: int):
    from imblearn.under_sampling import NearMiss
    nm = NearMiss(version=version)
    X_resampled_nm, y_resampled = nm.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled_nm, y_resampled, 'NearMiss' + str(version))


def tomek_links(X, y, sampling_strategy: str):
    from imblearn.under_sampling import TomekLinks
    tl = TomekLinks(sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = tl.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'TomekLinks')


def edited_nearest_neighbours(X, y, kind_sel: str):
    sorted(Counter(y).items())

    from imblearn.under_sampling import EditedNearestNeighbours
    enn = EditedNearestNeighbours(kind_sel=kind_sel)
    X_resampled, y_resampled = enn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'Edited Nearest Neighbours')


def repeated_edited_nearest_neighbours(X, y):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    renn = RepeatedEditedNearestNeighbours()
    X_resampled, y_resampled = renn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'Repeated Edited Nearest Neighbours')


def allknn(X, y):
    from imblearn.under_sampling import AllKNN
    allknn = AllKNN()
    X_resampled, y_resampled = allknn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'AllKNN')


def condensed_nearest_neighbours(X, y):
    from imblearn.under_sampling import CondensedNearestNeighbour
    cnn = CondensedNearestNeighbour(random_state=0)
    X_resampled, y_resampled = cnn.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'Condensed Nearest Neighbours')


def one_sided_selection(X, y):
    from imblearn.under_sampling import OneSidedSelection
    oss = OneSidedSelection(random_state=0)
    X_resampled, y_resampled = oss.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'One Sided Selection')


def neighbourhood_cleaning_rule(X, y):
    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule()
    X_resampled, y_resampled = ncr.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'Neighbourhood Cleaning Rule')


def instance_hardness_threshold(X, y):
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import InstanceHardnessThreshold
    iht = InstanceHardnessThreshold(random_state=0,
                                    estimator=LogisticRegression(
                                        solver='lbfgs', multi_class='auto'))
    X_resampled, y_resampled = iht.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'Instance Hardness Threshold')


if __name__ == '__main__':
    print("Pure clusterized data:")
    X, y = get_classified_data()
    print(sorted(Counter(y).items()))
    draw_plot(X, y, 'Pure data')

    print("ClusterClassification:")
    cluster_centroids(X, y)

    print("RandomUnderSampling:")
    random_under_sampler(X, y)

    print("NearMiss:")
    near_miss(X, y, version=1)

    print('TomekLinks:')
    tomek_links(X, y, sampling_strategy='auto')

    print('Edited Nearest Neighbours:')
    edited_nearest_neighbours(X, y, kind_sel='all')
    print('Repeated Edited Nearest Neighbours:')
    repeated_edited_nearest_neighbours(X, y)
    print('AllKNN:')
    allknn(X, y)

    print("Condensed Nearest Neighbours:")
    condensed_nearest_neighbours(X, y)

    print("One Sided Selection:")
    one_sided_selection(X, y)

    print("Neighbourhood Cleaning Rule")
    neighbourhood_cleaning_rule(X, y)

    print("Instance Hardness Threshold")
    instance_hardness_threshold(X, y)
