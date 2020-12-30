import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.balancing.data_controller import path_data_dir
import pandas as pd

path_balanced_csv = path_data_dir + "/balanced_csv_test"


def train_and_score(X, y, cores_count):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=cores_count)
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        print("Train error:", e)

    return clf.score(X_test, y_test)


def percent_change(original, value):
    return round((value - original) / original * 100, 2)


def resample_and_write_to_csv(obj, X, y, name):
    X_resampled, y_resampled = obj.fit_resample(X, y)

    write_df = X_resampled.append(y_resampled)
    write_df.to_csv(path_balanced_csv + "/" + name + ".csv", index=False)
    print("Balanced:", name)

    return X_resampled, y_resampled


def train_and_compare_all(score_original, cores_count):
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
