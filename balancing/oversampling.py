def random_over_sampler(X, y):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled


def smote(X, y, cores_count):
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE(n_jobs=cores_count).fit_resample(X, y)
    return X_resampled, y_resampled


def adasyn(X, y, cores_count):
    from imblearn.over_sampling import ADASYN
    X_resampled, y_resampled = ADASYN(n_jobs=cores_count).fit_resample(X, y)
    return X_resampled, y_resampled


def borderline_smote(X, y, cores_count):
    from imblearn.over_sampling import BorderlineSMOTE
    X_resampled, y_resampled = BorderlineSMOTE(n_jobs=cores_count).fit_resample(X, y)
    return X_resampled, y_resampled


def smotenc(X, y, cores_count):
    from imblearn.over_sampling import SMOTENC
    # Do poprawnego działania należy wprowadzić ich indeksy lub maskę boolowską za pomocą parametru categorical_features
    smote_nc = SMOTENC(random_state=0, n_jobs=cores_count)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    print(X_resampled[-5:])
    # Warto zauważyć że próbki wygenerowane dla pierwszej i ostatniej kolumny należą do tych samych kategorii które
    # były obecne przed balansowaniem (a nie potworzyły się jakieś mutanty).
    return X_resampled, y_resampled
