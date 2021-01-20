from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE


def adasyn_optimized():
    return ADASYN(n_jobs=-1)


def borderline_smote_optimized():
    return BorderlineSMOTE(n_jobs=-1)


def kmeans_smote_optimized():
    return KMeansSMOTE(n_jobs=-1)


def random_over_sampler_optimized():
    return RandomOverSampler(random_state=0)


def smote_optimized():
    return SMOTE(n_jobs=-1)


def smotenc_optimized(X):
    from imblearn.over_sampling import SMOTENC

    features_dict = dict(zip(list(X.columns), range(len(X.columns))))
    for key in ["click_timestamp", "nb_clicks_1week", "product_price"]:  # continue features list
        if key in features_dict.keys():
            features_dict.pop(key)

    return SMOTENC(categorical_features=list(features_dict.values()), k_neighbors=2, n_jobs=-1)


def svm_smote_optimized():
    return SVMSMOTE(n_jobs=-1)


def balance_all_oversampling(X):
    print("Oversampling methods comparsion:")

    random_over_sampler_optimized()
    smotenc_optimized(X)
    # adasyn_optimized(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # borderline_smote_optimized(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # kmeans_smote_optimized(X, y)# UWAGA! Generuje nowe dane dla cech kategorycznych!
    # smote_optimized(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
    # svm_smote_optimized(X, y)  # UWAGA! Generuje nowe dane dla cech kategorycznych!
