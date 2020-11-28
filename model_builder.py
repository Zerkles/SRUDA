#!/home/kwitnoncy/anaconda3/envs/inz/bin/python
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from collections import Counter

import numpy as np
import pandas as pd
from typing import List


def prepare_data(filename: str, 
                 separator: str = '',
                 useless_columns: List[str] = [],
                 labels_header: str = '', 
                 labels: List[int] = []
                 ) -> (int, np.ndarray, List[int]):
    data = None

    try:
        if separator:
            data = pd.read_csv(filename, sep=separator)
        else:
            data = pd.read_csv(filename, sep='\t')
    except FileNotFoundError:
        print('No file named: ' + filename)
        return (1, None, None)


    if data.empty:
        return (2, None, None)

    current_header = list(data.columns)

    for h in useless_columns:
        try:
            current_header.remove(h)
        except ValueError:
            print('No column named: ', h)
            return (3, None, None)

    if labels:
        # load labels to y
        pass
    elif labels_header:
        current_header.remove(labels_header)
        y = data[labels_header]
    else:
        return (4, None, None)

    X = data.loc[:, current_header]
    X = X.to_numpy()
    y = y.to_numpy()
    
    return (0, X, y)
    

def split(X: np.ndarray, y: List[int]) -> (np.ndarray, np.ndarray, List[int], List[int]):
    return train_test_split(X, y, test_size=0.2, random_state=7)


def teach_model(model_name: str, X: (np.ndarray, np.ndarray), y: (List[int], List[int])) -> float:
    model = None
    returning = False

    if model_name == 'XGB':
        returning = False
        model = XGBClassifier()
    elif model_name == 'log_reg':
        returning = True
        model = LogisticRegression(random_state=0, solver='sag', max_iter=10.000.000)
    elif model_name == 'tree':
        returning = True
        model = tree.DecisionTreeClassifier()



print(prepare_data(filename='data_100k.csv', separator='\t', useless_columns=['SalesAmountInEuro', 'time_delay_for_conversion'], labels_header="Sales"))

