from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE

from balancing.utilities import resample_and_write_to_csv


def random_over_sampler_variations(X, y):
    obj = RandomOverSampler(random_state=0)
    resample_and_write_to_csv(obj, X, y, "RandomOverSampler" + str(obj.get_params()))


def smote_variations(X, y, cores_count):
    obj = SMOTE(n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "SMOTE" + str(obj.get_params()))


def adasyn_variations(X, y, cores_count):
    obj = ADASYN(n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "ADASYN" + str(obj.get_params()))


def borderline_smote_variations(X, y, cores_count):
    obj = BorderlineSMOTE(n_jobs=cores_count)
    resample_and_write_to_csv(obj, X, y, "BorderlineSMOTE" + str(obj.get_params()))


def smotenc(X, y, cores_count):
    from imblearn.over_sampling import SMOTENC
    # Do poprawnego działania należy wprowadzić ich indeksy lub maskę boolowską za pomocą parametru categorical_features
    smote_nc = SMOTENC(random_state=0, n_jobs=cores_count)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    print(X_resampled[-5:])
    # Warto zauważyć że próbki wygenerowane dla pierwszej i ostatniej kolumny należą do tych samych kategorii które
    # były obecne przed balansowaniem (a nie potworzyły się jakieś mutanty).
    return X_resampled, y_resampled


def balance_all_oversampling(X, y, cores_count):
    print("Oversampling methods comparsion:")

    random_over_sampler_variations(X, y)
    smote_variations(X, y, cores_count)
    adasyn_variations(X, y, cores_count)
    borderline_smote_variations(X, y, cores_count)

    # describe_balancing(X, y, score_original, smotenc, "SMOTENC")    # TODO: tu coś się wywala
