from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier


# load data and shuffle observations

def test_on_data(X, y):

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2333)
    print 'train set: {}, test set: {}'.format(len(x_train),len(x_test))
    cls = XGBClassifier()
    cls.fit(x_train, y_train)
    # on test
    pred = cls.predict(x_test)
    print 'xgb accuracy score test', accuracy_score(y_test, pred)

    # on all
    pred = cls.predict(X)
    print 'xgb accuracy score all', accuracy_score(y, pred)

    # compare to gbrt in sklearn
    cls = GradientBoostingClassifier()
    cls.fit(x_train, y_train)
    # on test
    pred = cls.predict(x_test)
    print 'sklearn accuracy score test', accuracy_score(y_test, pred)

    # on all
    pred = cls.predict(X)
    print 'sklearn accuracy score all', accuracy_score(y, pred)


data = datasets.load_digits()
X = data.data
y = data.target
print 'test on mnist'
test_on_data(X, y)
print '====================================================='
data = datasets.load_iris()
X = data.data
y = data.target
print 'test on iris'
test_on_data(X, y)
print '====================================================='
data = datasets.fetch_olivetti_faces()
X = data.data
y = data.target
print 'test on olivetti faces'
test_on_data(X, y)
print '====================================================='
