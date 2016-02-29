from sklearn.cross_validation import cross_val_score
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf_pf = GaussianNB()
print np.unique(Y),'xxx'
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[-0.8, -1]]))
print cross_val_score(clf_pf,X,Y)
"""
Incremental fit on a batch of samples.
This method is expected to be called several times consecutively on different chunks of a dataset so as to implement out-of-core or online learning.
This is especially useful when the whole dataset is too big to fit in memory at once.
This method has some performance and numerical stability overhead, 
hence it is better to call partial_fit on chunks of data that are as large as possible 
(as long as fitting in the memory budget) to hide the overhead.
"""


iris = datasets.load_iris()
print iris.feature_names
print iris.data.size
print iris.target_names
print iris.target
print iris.target.size

clf = GaussianNB()
clf.fit(iris.data, iris.target)
print cross_val_score(clf,iris.data,iris.target)
