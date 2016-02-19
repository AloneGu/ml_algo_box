from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
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
        self.knn_test()
        self.svm_test()
        print '-----------------------'

    def dct_test(self):
        print 'dct test, depth', self.dct_depth
        start_time = time.time()
        dct_clf = DecisionTreeClassifier(max_depth=self.dct_depth)
        print cross_val_score(dct_clf, self.x_data, self.y_data)
        dct_clf.fit(self.x_data, self.y_data)
        print dct_clf.score(self.x_test, self.y_test), 'time cost', time.time() - start_time

    def knn_test(self):
        print 'knn test, n count', self.n_neighbores
        start_time = time.time()
        knn_clf = KNeighborsClassifier(n_neighbors=self.n_neighbores)
        print cross_val_score(knn_clf, self.x_data, self.y_data)
        knn_clf.fit(self.x_data, self.y_data)
        print knn_clf.score(self.x_test, self.y_test), 'time cost', time.time() - start_time

    def svm_test(self):
        print 'svm test'
        start_time = time.time()
        svm_clf = svm.SVC()
        print cross_val_score(svm_clf, self.x_data, self.y_data)
        svm_clf.fit(self.x_data, self.y_data)
        print svm_clf.score(self.x_test, self.y_test), 'time cost', time.time() - start_time


def get_wifi_x_y_data():
    bssid_cnt = {}
    place_set = set()
    f = open('wifi_data.txt')
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
    f = open('55_2015-11-12.csv')
    csv_reader = csv.reader(f)
    x_data, y_data, y_set = [], [], set()
    for row in csv_reader:
        x_data.append(map(int, row[:-1]))
        tmp_y = int(row[-1])
        y_data.append(tmp_y)
        y_set.add(tmp_y)
    return x_data, y_data, len(y_set)


if __name__ == '__main__':
    wifi_x, wifi_y, neighbor_num, _ = get_wifi_x_y_data()
    t = classifier_test(wifi_x, wifi_y, neighbor_num)
    t.run()
    wifi_class_x, wifi_class_y, neighbor_num = get_class_x_y_data()
    t = classifier_test(wifi_class_x, wifi_class_y, neighbor_num)
    t.run()
