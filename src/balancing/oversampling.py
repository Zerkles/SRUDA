from imblearn.over_sampling import RandomOverSampler, SMOTENC


def random_over_sampler_optimized():
    return RandomOverSampler(sampling_strategy='auto', random_state=0)


def smotenc_optimized(X_columns_names: list):
    features_dict = dict(zip(X_columns_names, range(len(X_columns_names))))
    for key in ["click_timestamp", "nb_clicks_1week", "product_price"]:  # continue features list
        if key in features_dict.keys():
            features_dict.pop(key)

    return SMOTENC(categorical_features=list(features_dict.values()), k_neighbors=10, n_jobs=-1)
