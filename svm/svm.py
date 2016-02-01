from sklearn import svm

def simple_example():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)
    print clf.predict([[2., 2.]])
    # get support vectors
    print clf.support_vectors_

    # get indices of support vectors
    print clf.support_

    # get number of support vectors for each class
    print clf.n_support_


simple_example()

