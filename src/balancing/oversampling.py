from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

from src.balancing.utilities import resample_and_write_to_csv, feature_graph_generator


def adasyn_variations(X, y):
    obj = ADASYN(n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "ADASYN" + str(obj.get_params()))


def borderline_smote_variations(X, y):
    obj = BorderlineSMOTE(n_jobs=-1)
    X_resampled, y_resampled = resample_and_write_to_csv(obj, X, y, "BorderlineSMOTE" + str(obj.get_params()))
    print(len(X["product_id"].unique()), len(X_resampled["product_id"].unique()))


def kmeans_smote_variations(X, y):
    obj = KMeansSMOTE(n_jobs=-1)
    X_resampled, y_resampled = resample_and_write_to_csv(obj, X, y, "KMeansSMOTE" + str(obj.get_params()))
    print(len(X["product_id"].unique()), len(X_resampled["product_id"].unique()))


def random_over_sampler_variations(X, y):
    obj = RandomOverSampler(random_state=0)
    resample_and_write_to_csv(obj, X, y, obj.__str__())


def smote_variations(X, y):
    obj = SMOTE(n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "SMOTE" + str(obj.get_params()))


def smotenc_variations(X, y):
    from imblearn.over_sampling import SMOTENC

    features_dict = dict(zip(list(X.columns), range(len(X.columns))))
    for key in ["click_timestamp", "nb_clicks_1week", "product_price"]:  # continue features list
        if key in features_dict.keys():
            features_dict.pop(key)

    # obj = SMOTENC(random_state=RANDOM_STATE, n_jobs=-1, categorical_features=list(features_dict.values()))
    #
    # values_dict = {"k_neighbors": list(range(1, 20))}
    # gridsearch_with_graph(obj, values_dict, X, y)


def svm_smote_variations(X, y):
    obj = SVMSMOTE(n_jobs=-1)
    resample_and_write_to_csv(obj, X, y, "SVMSMOTE" + str(obj.get_params()))


def balance_all_oversampling(X, y):
    print("Oversampling methods comparsion:")

    random_over_sampler_variations(X, y)
    smotenc_variations(X, y)
    # adasyn_variations(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # borderline_smote_variations(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # kmeans_smote_variations(X, y)# UWAGA! Generuje nowe dane dla cech kategorycznych!
    # smote_variations(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # svm_smote_variations(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
