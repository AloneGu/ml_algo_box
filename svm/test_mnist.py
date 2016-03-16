from sklearn.svm import SVC
from sklearn import metrics

def get_mnist_data():
    import gzip,pickle
    f = gzip.open('../data/mnist.pkl.gz', "rb")
    train, val, test = pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = get_mnist_data()
print 'get data'
model = SVC(kernel='rbf', probability=True)
model.fit(train_x, train_y)
predict = model.predict(test_x)
print 'get model'
print metrics.accuracy_score(test_y,predict)
print metrics.recall_score(test_y,predict)