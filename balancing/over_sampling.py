from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data import data_controller
from examples.under_sampling_examples import draw_plot, get_classified_data


def random_over_sampler(X, y):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    from collections import Counter
    print(sorted(Counter(y_resampled).items()))

    return X_resampled, y_resampled


def smote(X, y):
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def adasyn(X, y):
    from imblearn.over_sampling import ADASYN
    X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def borderline_smote(X, y):
    from imblearn.over_sampling import BorderlineSMOTE
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


def smotenc(X, y):
    from imblearn.over_sampling import SMOTENC
    # Do poprawnego działania należy wprowadzić ich indeksy lub maskę boolowską za pomocą parametru categorical_features
    smote_nc = SMOTENC(random_state=0)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))

    print(X_resampled[-5:])
    # Warto zauważyć że próbki wygenerowane dla pierwszej i ostatniej kolumny należą do tych samych kategorii które
    # były obecne przed balansowaniem (a nie potworzyły się jakieś mutanty).
    return X_resampled, y_resampled


def train_and_score(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))


def compare_balancing_methods(X, y):
    print("Random Over Sampler:")
    X_resampled, y_resampled = random_over_sampler(X, y)
    train_and_score(X_resampled, y_resampled)

    print("SMOTE:")
    smote(X, y)
    train_and_score(X_resampled, y_resampled)

    print("ADASYN:")
    adasyn(X, y)
    train_and_score(X_resampled, y_resampled)

    print("Borderline SMOTE:")
    borderline_smote(X, y)
    train_and_score(X_resampled, y_resampled)

    # print("SMOTENC:")
    # smotenc(X, y)
    # train_and_score(X_resampled, y_resampled)
    # TODO: tu coś się wywala


if __name__ == '__main__':
    samples_count = 100000
    X_original, y_original = data_controller.get_converted_data(samples_count)
    print(sorted(Counter(y_original).items()))

    print("Score for not balanced data:")
    train_and_score(X_original, y_original)

    compare_balancing_methods(X_original, y_original)
