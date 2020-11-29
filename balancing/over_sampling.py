# imbalanced-learn library
from train_score_test import train_and_score, percent_change


def random_over_sampler(X, y):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled


def smote(X, y):
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    return X_resampled, y_resampled


def adasyn(X, y):
    from imblearn.over_sampling import ADASYN
    X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    return X_resampled, y_resampled


def borderline_smote(X, y):
    from imblearn.over_sampling import BorderlineSMOTE
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
    return X_resampled, y_resampled


def smotenc(X, y):
    from imblearn.over_sampling import SMOTENC
    # Do poprawnego działania należy wprowadzić ich indeksy lub maskę boolowską za pomocą parametru categorical_features
    smote_nc = SMOTENC(random_state=0)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    print(X_resampled[-5:])
    # Warto zauważyć że próbki wygenerowane dla pierwszej i ostatniej kolumny należą do tych samych kategorii które
    # były obecne przed balansowaniem (a nie potworzyły się jakieś mutanty).
    return X_resampled, y_resampled


def compare_oversampling_methods(X, y, score_original):
    print("Random Over Sampler:")
    X_resampled, y_resampled = random_over_sampler(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("SMOTE:")
    X_resampled, y_resampled = smote(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("ADASYN:")
    X_resampled, y_resampled = adasyn(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    print("Borderline SMOTE:")
    X_resampled, y_resampled = borderline_smote(X, y)
    score = train_and_score(X_resampled, y_resampled)
    print("Score:", score)
    print("Percent Change:", percent_change(score_original, score), "\n")

    # print("SMOTENC:")
    # smotenc(X, y)
    # train_and_score(X_resampled, y_resampled)
    # TODO: tu coś się wywala
