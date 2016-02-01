from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
x=cross_val_score(clf, iris.data, iris.target)
print x

from util import get_x_y_data
x_data,y_data,zone_cnt,zone_int_dict = get_x_y_data()
clf_new = DecisionTreeClassifier(random_state=0)
print cross_val_score(clf_new,x_data,y_data)