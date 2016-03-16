from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time
import numpy as np


class classifier_test(object):
    def __init__(self, x_data, y_data, n_neighbors=3, dct_depth=5):
        # x_data: list of x data, y data : tags for each feature
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.n_neighbores = n_neighbors
        self.dct_depth = dct_depth
        _, self.x_test, _, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.6, random_state=0)

    def run(self):
        self.dct_test()
        print '-----------------------'
        self.knn_test()
        print '-----------------------'
        self.svm_test()
        print '-----------------------'
        self.gaussian_bayes_test()
        print '-----------------------'
        self.logistic_regression_classifier()
        print '-----------------------'
        self.random_forest_classifier()
        print '-----------------------'
        self.gradient_boosting_classifier()
        print '-----------------------'

    def dct_test(self):
        print 'dct test, depth', self.dct_depth
        start_time = time.time()
        dct_clf = DecisionTreeClassifier(max_depth=self.dct_depth)
        print 'cross validation score',cross_val_score(dct_clf, self.x_data, self.y_data)
        dct_clf.fit(self.x_data, self.y_data)
        print 'score',dct_clf.score(self.x_test, self.y_test)
        print 'time cost', time.time() - start_time

    def knn_test(self):
        print 'knn test, n count', self.n_neighbores
        start_time = time.time()
        knn_clf = KNeighborsClassifier(n_neighbors=self.n_neighbores)
        print 'cross validation score',cross_val_score(knn_clf, self.x_data, self.y_data)
        knn_clf.fit(self.x_data, self.y_data)
        print 'score',knn_clf.score(self.x_test, self.y_test)
        print 'time cost', time.time() - start_time

    def svm_test(self):
        print 'svm test'
        start_time = time.time()
        svm_clf = svm.SVC()
        print 'cross validation score',cross_val_score(svm_clf, self.x_data, self.y_data)
        svm_clf.fit(self.x_data, self.y_data)
        print 'score',svm_clf.score(self.x_test, self.y_test)
        print 'time cost', time.time() - start_time

    def gaussian_bayes_test(self):
        print 'gaussian bayes test'
        start_time = time.time()
        g_bayes_clf = GaussianNB()
        print 'cross validation score',cross_val_score(g_bayes_clf, self.x_data, self.y_data)
        g_bayes_clf.fit(self.x_data, self.y_data)
        print 'score',g_bayes_clf.score(self.x_test, self.y_test)
        print 'time cost', time.time() - start_time

    # Logistic Regression Classifier
    def logistic_regression_classifier(self):
        model = LogisticRegression(penalty='l2')
        model.fit(self.x_data, self.y_data)
        print 'logistic cross validation score',cross_val_score(model, self.x_data, self.y_data)

    # Random Forest Classifier
    def random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=8)
        model.fit(self.x_data, self.y_data)
        print 'random forest cross validation score',cross_val_score(model, self.x_data, self.y_data)

    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self):
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(self.x_data, self.y_data)
        print 'GBDT cross validation score',cross_val_score(model, self.x_data, self.y_data)


def get_wifi_x_y_data():
    bssid_cnt = {}
    place_set = set()
    f = open('../data/wifi_data.txt')
    content = f.readlines()
    x_data, y_data = [], []
    zone_int_dict = {}
    start_int = 0

    for row in content:
        first_colon_index = row.find(':')
        data = eval(row[first_colon_index + 1:])
        for k in data.keys():
            if k not in bssid_cnt:
                bssid_cnt[k] = 0
            bssid_cnt[k] += 1

    import operator
    sorted_result = sorted(bssid_cnt.items(), key=operator.itemgetter(1))
    # print 'sort keys', sorted_result
    bssid_list = [x[0] for x in sorted_result[-3:]]

    for row in content:
        first_colon_index = row.find(':')
        place = row[:first_colon_index]
        place_set.add(place)
        data = eval(row[first_colon_index + 1:])
        for k in data.keys():
            if k not in bssid_list:
                del (data[k])
        for bssid in bssid_list:
            if bssid not in data:
                data[bssid] = '-100'
        tmp_x_data = [int(data[bssid]) for bssid in bssid_list]
        if place not in zone_int_dict:
            zone_int_dict[place] = start_int
            start_int += 1
        tmp_y_data = zone_int_dict[place]
        y_data.append(tmp_y_data)
        x_data.append(tmp_x_data)
    return x_data, y_data, len(place_set), zone_int_dict


def get_class_x_y_data():
    import csv
    f = open('../data/108_wifi_training.csv')
    csv_reader = csv.reader(f)
    x_data, y_data, y_set = [], [], set()
    for row in csv_reader:
        x_data.append(map(int, row[:-1]))
        tmp_y = int(row[-1])
        y_data.append(tmp_y)
        y_set.add(tmp_y)
    return x_data, y_data, len(y_set)


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


if __name__ == '__main__':
    #wifi_x, wifi_y, neighbor_num, _ = get_wifi_x_y_data()
    #t = classifier_test(wifi_x, wifi_y, neighbor_num)
    #t.run()
    wifi_class_x, wifi_class_y, neighbor_num = get_class_x_y_data()
    t = classifier_test(wifi_class_x, wifi_class_y, neighbor_num)
    t.run()
