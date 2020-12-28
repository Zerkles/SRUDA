from collections import Counter

from examples.under_sampling_examples import draw_plot, get_classified_data

def linear_svc(X_resampled, y_resampled, balancing_method_name: str):
    from sklearn.svm import LinearSVC
    clf = LinearSVC()
    clf.fit(X_resampled, y_resampled)  # doctest : +ELLIPSIS
    y_trained = clf.decision_function(X_resampled)
    print(X_resampled, y_trained)
    # draw_plot(X_resampled, y_trained, "Linear SVC for " + balancing_method_name) TODO: Nie działa


def random_over_sampler(X, y):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    from collections import Counter
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, 'Random Over Sampler')

    print("    for heterogenous data:")
    import numpy as np
    X_hetero = np.array([['xxx', 1, 1.0], ['yyy', 2, 2.0], ['zzz', 3, 3.0]],
                        dtype=np.object)
    y_hetero = np.array([0, 0, 1])
    X_resampled2, y_resampled2 = ros.fit_resample(X_hetero, y_hetero)
    print(X_resampled2)
    print(y_resampled2)

    return X_resampled, y_resampled


def smote(X, y):
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, "SMOTE")


def adasyn(X, y):
    from imblearn.over_sampling import ADASYN
    X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, "ADASYN")


def borderline_smote(X, y):
    from imblearn.over_sampling import BorderlineSMOTE
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    draw_plot(X_resampled, y_resampled, "Borderline SMOTE")


def smotenc(X, y):
    import numpy as np
    # create a synthetic data set with continuous and categorical features
    rng = np.random.RandomState(42)
    n_samples = 50
    X = np.empty((n_samples, 3), dtype=object)
    X[:, 0] = rng.choice(['A', 'B', 'C'], size=n_samples).astype(object)
    X[:, 1] = rng.randn(n_samples)
    X[:, 2] = rng.randint(3, size=n_samples)
    y = np.array([0] * 20 + [1] * 30)
    print(sorted(Counter(y).items()))

    # W tym utworzonym zbiorze pierwsza i ostatnia cecha mogą być rozważane jako kateogrie.
    # Do poprawnego działania należy wprowadzić ich indeksy lub maskę boolowską za pomocą parametru categorical_features

    from imblearn.over_sampling import SMOTENC
    smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))

    print(X_resampled[-5:])
    draw_plot(X_resampled, y_resampled, "SMOTENC")
    # TODO: zbugowany ten wykres jest
    # Warto zauważyć że próbki wygenerowane dla pierwszej i ostatniej kolumny należą do tych samych kategorii które
    # były obecne przed balansowaniem (a nie potworzyły się jakieś mutanty).


if __name__ == '__main__':
    print("Pure clusterized data:")
    X, y = get_classified_data()
    print(y)
    print(X[0][1])
    print(sorted(Counter(y).items()))
    draw_plot(X, y, 'Pure data')

    # print("Random Over Sampler:")
    # X_resampled, y_resampled = random_over_sampler(X, y)
    # # linear_svc(x, y, "Random Over Sampler")
    #
    # print("SMOTE:")
    # smote(X, y)
    #
    # print("ADASYN:")
    # adasyn(X, y)
    #
    # print("Borderline SMOTE:")
    # borderline_smote(X, y)
    #
    # print("SMOTENC:")
    # smotenc(X, y)
