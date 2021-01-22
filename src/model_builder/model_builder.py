#!/home/kwitnoncy/anaconda3/envs/inz/bin/python
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from typing import List
import time


class ModelBuilder:
    def __init__(self, model_name: str,
                 filename: str,
                 unbalanced_filename: str,
                 separator: str = '',
                 useless_columns=None,
                 labels_header: str = '',
                 labels=None):
        self.model_name = model_name
        self.filename = filename
        self.unbalanced_filename = unbalanced_filename
        self.separator = separator
        self.useless_columns = useless_columns
        self.labels_header = labels_header
        self.labels = labels

    def get_result(self) -> (dict, List[int], List[int], List[int], List[int]):
        error, x, y, x_un, y_un = self.prepare_data()
        if error:
            print('Something went wrong', error)
            return error

        x_train, x_test, y_train, y_test = ModelBuilder.split(x, y)

        return self.teach_model(x=(x_train, x_test, x_un), y=(y_train, y_test, y_un))

    def prepare_data(self) -> (int, np.ndarray, List[int]):
        try:
            if self.separator:
                balanced_data = pd.read_csv(self.filename, sep=self.separator)
                unbalanced_data = pd.read_csv(self.unbalanced_filename, sep=self.separator)
            else:
                balanced_data = pd.read_csv(self.filename, sep=',')
                unbalanced_data = pd.read_csv(self.unbalanced_filename, sep=',')
        except FileNotFoundError:
            print('No file named: ' + self.filename)
            return 1, None, None

        if balanced_data.empty or unbalanced_data.empty:
            return 2, None, None

        current_header = list(balanced_data.columns)

        if self.labels:
            y = self.labels
        elif self.labels_header:
            current_header.remove(self.labels_header)
            y = balanced_data[self.labels_header]
        else:
            return 4, None, None

        y_un = unbalanced_data[self.labels_header]
        if len(y_un) == 0:
            return 5, None, None

        X = balanced_data.loc[:, current_header]
        X = X.to_numpy()
        X[np.isnan(X)] = 0.0
        y = y.to_numpy()

        X_un = unbalanced_data.loc[:, current_header]
        X_un = X_un.to_numpy()
        X_un[np.isnan(X_un)] = 0.0
        y_un = y_un.to_numpy()

        return 0, X, y, X_un, y_un

    def teach_model(self,
                    x: (np.ndarray, np.ndarray, np.ndarray),
                    y: (List[int], List[int], List[int])
                    ) -> (dict, List[int], List[int], List[int], List[int]):
        result = {}

        if self.model_name == 'xgb':
            model = XGBClassifier(subsample=0.25,
                                  n_estimators=270,
                                  min_child_weight=3,
                                  max_depth=7,
                                  max_delta_step=4,
                                  learning_rate=0.19,
                                  gamma=6,
                                  verbosity=0,
                                  booster='gbtree',
                                  n_jobs=-1,
                                  use_label_encoder=False)
            result['model'] = 'xgb'
        elif self.model_name == 'cat':
            model = CatBoostClassifier(loss_function='Logloss',
                                       scale_pos_weight=24.0,
                                       reg_lambda=9.5,
                                       n_estimators=80,
                                       max_depth=6,
                                       learning_rate=0.075,
                                       border_count=165,
                                       verbose=0)
            result['model'] = 'cat'
        elif self.model_name == 'reg':
            model = LogisticRegression(tol=0.001,
                                       solver='newton-cg',
                                       penalty='l2',
                                       max_iter=1000,
                                       intercept_scaling=1.0,
                                       fit_intercept=True,
                                       dual=False,
                                       class_weight='balanced',
                                       C=0.0001,
                                       verbose=1000)
            result['model'] = 'reg'
        elif self.model_name == 'tree':
            model = DecisionTreeClassifier(splitter='random',
                                           random_state=25,
                                           min_weight_fraction_leaf=0.3,
                                           min_samples_split=7,
                                           max_features='auto',
                                           max_depth=9,
                                           class_weight='balanced',
                                           criterion='gini')
            result['model'] = 'tree'
        elif self.model_name == 'for':
            model = RandomForestClassifier(random_state=25,
                                           min_weight_fraction_leaf=0.3,
                                           min_samples_split=7,
                                           max_features='auto',
                                           max_depth=9,
                                           class_weight='balanced',
                                           criterion='gini')
            result['model'] = 'RandomForest'
        else:
            return {}, [], []

        time_start = time.time()
        model.fit(x[0], y[0])
        result['train_time'] = time.time() - time_start

        # balanced data
        time_start = time.time()
        predicted_balanced = model.predict(x[1])
        result['balanced'] = {}
        result['balanced']['test_time'] = time.time() - time_start
        result['balanced']['mean_score'] = model.score(x[1], y[1])
        result['balanced']['predict_proba'] = model.predict_proba(x[1])
        result['balanced']['predicted'] = predicted_balanced
        result['balanced']['real'] = y[1]

        result['balanced']['TN'], result['balanced']['FP'], result['balanced']['FN'], result['balanced']['TP'] = \
            ModelBuilder.create_confusion_table(
                predicted=predicted_balanced,
                real=y[1])

        # unbalanced data
        time_start = time.time()
        predicted_unbalanced = model.predict(x[2])
        result['unbalanced'] = {}
        result['unbalanced']['test_time'] = time.time() - time_start
        result['unbalanced']['mean_score'] = model.score(x[2], y[2])
        result['unbalanced']['predict_proba'] = model.predict_proba(x[2])
        result['unbalanced']['predicted'] = predicted_unbalanced
        result['unbalanced']['real'] = y[2]

        result['unbalanced']['TN'], result['unbalanced']['FP'], result['unbalanced']['FN'], result['unbalanced']['TP'] \
            = ModelBuilder.create_confusion_table(
            predicted=predicted_unbalanced,
            real=y[2])

        return result, predicted_balanced, y[1], predicted_unbalanced, y[2]

    @staticmethod
    def create_confusion_table(predicted: List[int], real: List[int]) -> (int, int, int, int):
        return confusion_matrix(real, predicted).ravel()

    @staticmethod
    def split(x: np.ndarray, y: List[int]) -> (np.ndarray, np.ndarray, List[int], List[int]):
        return train_test_split(x, y, test_size=0.2, random_state=7)


def multiple_models_single_data(models: List, data: List[List[int]]) -> List[int]:
    print(models)
    print(data)
    return [0, 1, 0, 1]
