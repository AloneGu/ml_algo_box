X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))

from util import get_x_y_data

TEST_DATA_ROWS = 20
import numpy as np

def train_data():
    x_data,y_data,zone_cnt,zone_int_dict = get_x_y_data()

    knn = KNeighborsClassifier()

    indices = np.random.permutation(len(x_data))
    x_train = x_data
    y_train = y_data
    x_test  = x_data[indices[-TEST_DATA_ROWS:]]
    y_test  = y_data[indices[-TEST_DATA_ROWS:]]
    knn.fit(x_train, y_train) # start training
    print 'training data count:',len(indices), ' number of zones:',zone_cnt
    test_result = knn.predict(x_test) # test
    prob_test_result = knn.predict_proba(x_test)
    print prob_test_result

    # no duplicate value, so reverse this dictionary
    int_zone_dict = dict(zip(zone_int_dict.values(), zone_int_dict.keys()))

    print 'predict result:',test_result,[int_zone_dict[x] for x in test_result] # test result
    print 'ground truth:',y_test,[int_zone_dict[x] for x in y_test] # ground truth
    cnt = 0
    for i in range(TEST_DATA_ROWS):
        if test_result[i] == y_test[i]:
            cnt+=1
    print 'accurate rate',cnt*1.0/TEST_DATA_ROWS

    from sklearn.cross_validation import cross_val_score
    print cross_val_score(knn,x_train,y_train)

train_data()
