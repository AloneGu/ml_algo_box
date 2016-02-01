import numpy as np
from ..util import get_x_y_data
from sklearn.cluster import KMeans
TEST_DATA_ROWS = 20

#class sklearn.cluster.KMeans
#(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)

x_data,y_data,zone_cnt,zone_int_dict = get_x_y_data()
# no duplicate value, so reverse this dictionary
int_zone_dict = dict(zip(zone_int_dict.values(), zone_int_dict.keys()))

kmeans = KMeans(n_clusters=zone_cnt) # a,b,c,d,e  5 centor
kmeans.fit(x_data)
print kmeans.get_params()
# centers
print kmeans.cluster_centers_

# every lable for cluster
print kmeans.labels_

# the smaller inertia is, the better the classifier works
print kmeans.inertia_

indices = np.random.permutation(len(x_data))
x_test  = x_data[indices[-TEST_DATA_ROWS:]]
x_distance = kmeans.transform(x_test)
test_result = kmeans.predict(x_test) # test

for type,dis in zip(test_result,x_distance):
    print type,dis

from sklearn.cross_validation import cross_val_score
print cross_val_score(kmeans,x_data)

