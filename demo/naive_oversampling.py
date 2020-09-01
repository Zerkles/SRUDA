from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.svm import LinearSVC
import numpy as np

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
    n_redundant=0, n_repeated=0, n_classes=3,
    n_clusters_per_class=1,
    weights=[0.01, 0.05, 0.94],
    class_sep=0.8, random_state=0)
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))

clf = LinearSVC()
clf.fit(X_resampled, y_resampled)

X_hetero = np.array([['xxx', 1, 1.0], ['yyy', 2, 2.0], ['zzz', 3, 3.0]], dtype=np.object)
y_hetero = np.array([0, 0, 1])
X_resampled, y_resampled = ros.fit_resample(X_hetero, y_hetero)

print(X_resampled)
print(y_resampled)

