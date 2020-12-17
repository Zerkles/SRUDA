
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.svm import LinearSVC




class Features:
    """description of class"""

    def select_fetures_select_from_model_LR( X,y,columns,iteration):
        selection = SelectFromModel(estimator=LogisticRegression(max_iter=iteration)).fit(X,y)
        selected_features = np .array( columns )[ selection . get_support () ]
        (" Features selected by SelectFromModel from logistic regression : " )
        print ( selected_features )        print ("")    def select_fetures_select_from_model_linearsvc( X,y,columns,iteration):
        selection = SelectFromModel(estimator=LinearSVC()).fit(X,y)
        selected_features = np .array( columns )[ selection . get_support () ]
        (" Features selected by SelectFromModel from Linear SVC : " )
        print ( selected_features )        print ("")    def select_fetures_select_from_model_lasso( X,y,columns,iteration):
        clf = LassoCV().fit(X, y)
        importance = np.abs(clf.coef_)
        print("importance")
        print(importance)
        idx_third = importance.argsort()[-4]
        threshold = importance[idx_third] + 0.00000001
        idx_features = (-importance).argsort()[:3]
        name_features = np.array(columns)[idx_features]
        print('Selected features: {}'.format(name_features))
        sfm = SelectFromModel(clf, threshold=threshold)
        sfm.fit(X, y)
        X_transform = sfm.transform(X)
        n_features = sfm.transform(X).shape[1]