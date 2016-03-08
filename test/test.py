c = open('test.txt').readlines()
test_x = []
for r in c:
    x = eval(r[:-2])
    test_x.append(x)

def get_class_x_y_data():
    import csv
    f = open('../data/55_2015-11-12.csv')
    csv_reader = csv.reader(f)
    x_data, y_data, y_set = [], [], set()
    for row in csv_reader:
        x_data.append(map(int, row[:-1]))
        tmp_y = int(row[-1])
        y_data.append(tmp_y)
        y_set.add(tmp_y)
    return x_data, y_data, len(y_set)

x_data,y_data,_ = get_class_x_y_data()
from sklearn import svm
svm_clf = svm.SVC()
import numpy
x_data = numpy.array(x_data)
y_data = numpy.array(y_data)
svm_clf.fit(x_data, y_data)
print svm_clf.predict(test_x)

from sklearn.naive_bayes import GaussianNB
g_bayes_clf = GaussianNB()
g_bayes_clf.fit(x_data,y_data)
print g_bayes_clf.predict(test_x)