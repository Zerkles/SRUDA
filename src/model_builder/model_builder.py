#!/home/kwitnoncy/anaconda3/envs/inz/bin/python
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from typing import List
import time


class ModelBuilder:
    def __init__(self, model_name: str,
                 filename: str,
                 separator: str = '',
                 useless_columns=None,
                 labels_header: str = '',
                 labels=None):
        self.model_name = model_name
        self.filename = filename
        self.separator = separator
        self.useless_columns = useless_columns
        self.labels_header = labels_header
        self.labels = labels

    def get_result(self) -> (dict, List[int], List[int]):
        # TODO repair return
        error, x, y, = self.prepare_data()
        if error:
            print('Something went wrong', error)
            return error

        x_train, x_test, y_train, y_test = ModelBuilder.split(x, y)

        return self.teach_model(x=(x_train, x_test), y=(y_train, y_test))

    def prepare_data(self) -> (int, np.ndarray, List[int]):
        if self.useless_columns is None:
            useless_columns = []
        if self.labels is None:
            labels = []

        try:
            if self.separator:
                data = pd.read_csv(self.filename, sep=self.separator)
            else:
                data = pd.read_csv(self.filename, sep='\t')
        except FileNotFoundError:
            print('No file named: ' + self.filename)
            return 1, None, None

        if data.empty:
            return 2, None, None

        current_header = list(data.columns)

        if self.useless_columns:
            for h in self.useless_columns:
                try:
                    current_header.remove(h)
                except ValueError:
                    print('No column named: ', h)
                    return 3, None, None

        if self.labels:
            y = self.labels
        elif self.labels_header:
            current_header.remove(self.labels_header)
            y = data[self.labels_header]
        else:
            return 4, None, None

        X = data.loc[:, current_header]
        X = X.to_numpy()
        X[np.isnan(X)] = 0.0
        y = y.to_numpy()

        return 0, X, y

    def teach_model(self, x: (np.ndarray, np.ndarray), y: (List[int], List[int])) -> (dict, List[int], List[int]):
        result = {}
        model = None

        if self.model_name == 'xgb':
            model = XGBClassifier()
            result['model'] = 'xgb'
        elif self.model_name == 'cat':
            model = CatBoostClassifier(iterations=2, depth=2, learning_rate=1, loss_function='Logloss', verbose=True)
            result['model'] = 'cat'
        elif self.model_name == 'reg':
            model = LogisticRegression(random_state=0, solver='sag', max_iter=10000000)
            result['model'] = 'reg'
        elif self.model_name == 'tree':
            model = tree.DecisionTreeClassifier()
            result['model'] = 'tree'

        if not model:
            return {}, [], []

        time_start = time.time()
        model.fit(x[0], y[0])
        result['train_time'] = time.time() - time_start

        time_start = time.time()
        predicted = model.predict(x[1])
        result['test_time'] = time.time() - time_start
        result['mean_score'] = model.score(x[1], y[1])
        result['predict_proba'] = model.predict_proba(x[1])

        result['TP'], result['FN'], result['TN'], result['FP'] = ModelBuilder.create_confusion_table(
            predicted=predicted,
            real=y[1])

        return result, x[1], predicted

    @staticmethod
    def create_confusion_table(predicted: List[int], real: List[int]) -> (int, int, int, int):
        tp = tn = fp = fn = 0
        for i in range(len(predicted)):
            if real[i] == 1:
                if predicted[i] == real[i]:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted[i] == real[i]:
                    tn += 1
                else:
                    fp += 1
        return tp, fn, tn, fp

    @staticmethod
    def split(x: np.ndarray, y: List[int]) -> (np.ndarray, np.ndarray, List[int], List[int]):
        return train_test_split(x, y, test_size=0.2, random_state=7)
