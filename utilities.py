import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from balancing.oversampling import *
from balancing.undersampling import *
from balancing.multi_imbalanced import *
from data.data_controller import path_data_dir
import pandas as pd

path_balanced_csv = path_data_dir + "/balanced_csv"


def train_and_score(X, y, cores_count):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=cores_count)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


def percent_change(original, value):
    return round((value - original) / original * 100, 2)


def balance_to_csv(x_y_pair, name: str):
    X_resampled, y_resampled = x_y_pair
    # del [x_y_pair]
    X_resampled["Sales"] = y_resampled
    X_resampled.to_csv(path_balanced_csv + "/" + name + ".csv")
    print("Balanced:", name)


def balance_train_comparsion(score_original, cores_count):
    for filename in os.listdir(path_balanced_csv):
        path_file = path_balanced_csv + '/' + filename
        print("Training:", filename[:-4])
        df = pd.read_csv(
            filepath_or_buffer=path_file,
            sep=',',
            index_col=0,
            low_memory=False)

        X_resampled = df[list(df.columns)[:-1]]
        y_resampled = df["Sales"]
        score = train_and_score(X_resampled.values, y_resampled.values, cores_count)

        print("Score:", score)
        print("Percent Change:", percent_change(score_original, score), "[%]\n")


def describe_balancing_multiclass(X, y, score_original, func, name: str):
    from multi_imbalance.utils.plot import plot_visual_comparision_datasets
    import matplotlib.pyplot as plt
    from collections import Counter
    print(name + ":")
    X_resampled, y_resampled = func(X, y)

    # print(sorted(Counter(y_resampled).items()))

    plot_visual_comparision_datasets(X, y, X_resampled, y_resampled, 'CriteoCS', 'Resampled CriteoCS with ' + name)
    plt.show()
    # plt.savefig("/graphs"+name+".png")

    score = train_and_score(X_resampled.values, y_resampled.values)

    X_resampled["Sales"] = y_resampled
    X_resampled.to_csv("balancing_csv/" + name + ".csv")

    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "[%]\n")


def balance_to_csv_all_undersampling(X, y, cores_count):
    print("Undersampling methods balancing:")
    # balance_to_csv(cluster_centroids(X, y, cores_count),
    #                "ClusterCentroids_5M")  # limited cores due to high memory demands

    balance_to_csv(condensed_nearest_neighbours(X, y, cores_count), "CondensedNearestNeighbours")

    # already proceeded
    # balance_to_csv(random_under_sampler(X, y), "RandomUnderSampler")
    # balance_to_csv(one_sided_selection(X, y, cores_count), "OneSidedSelection")
    # balance_to_csv(neighbourhood_cleaning_rule(X, y, cores_count), "NeighbourhoodCleaningRule")
    # balance_to_csv(near_miss(X, y, cores_count, version=1), "NearMiss")
    # balance_to_csv(tomek_links(X, y, cores_count, sampling_strategy='auto'), "TomekLinks")
    # balance_to_csv(edited_nearest_neighbours(X, y, cores_count, kind_sel='all'),
    #                "EditedNearestNeighbours")
    # balance_to_csv(repeated_edited_nearest_neighbours(X, y, cores_count),
    #                "RepeatedEditedNearestNeighbours")
    # balance_to_csv(allknn(X, y, cores_count), "AllKNN")

    # balance_to_csv(X, y, score_original, instance_hardness_threshold, "InstanceHardnessThreshold")
    # TODO: tu coś się wywala


def balance_to_csv_all_oversampling(X, y, cores_count):
    print("Oversampling methods comparsion:")

    balance_to_csv(random_over_sampler(X, y), "RandomOverSampler")
    balance_to_csv(smote(X, y, cores_count), "SMOTE")
    balance_to_csv(adasyn(X, y, cores_count), "ADASYN")
    balance_to_csv(borderline_smote(X, y, cores_count), "Borderline SMOTE")

    # describe_balancing(X, y, score_original, smotenc, "SMOTENC")
    # TODO: tu coś się wywala


def compare_multi_class_methods(X, y, score_original):
    import warnings
    warnings.filterwarnings("ignore")
    print("Multi-imbalanced methods comparsion:")

    describe_balancing_multiclass(X, y, score_original, global_cs, "GlobalCs")
    describe_balancing_multiclass(X, y, score_original, mdo, "MDO")
    describe_balancing_multiclass(X, y, score_original, soup, "SOUP")
    describe_balancing_multiclass(X, y, score_original, spider3, "SPIDER3")
    describe_balancing_multiclass(X, y, score_original, static_smote, "StaticSMOTE")
