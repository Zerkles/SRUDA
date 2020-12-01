from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_score(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


def percent_change(original, value):
    return round((value - original) / original * 100, 2)
