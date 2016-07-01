import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import csv
SRC_FILE = '../data/125_2016-04-13_occupancy_minutes.csv'

def load_data(f):
    csv_r = csv.reader(open(f))
    csv_r.next() # jump header
    x,y=[],[]
    for r in csv_r:
        tmp_t = datetime.datetime.strptime(r[1],'%Y-%m-%d %H:%M:%S')
        hour = tmp_t.hour*1.0
        minutes = tmp_t.minute*1.0
        v_occ_min = float(r[2])
        w_occ_min = float(r[3])
        v_occ = float(r[4])
        _y = float(r[-1])
        x.append([hour,minutes,v_occ_min,w_occ_min,v_occ])
        y.append(_y)
    return np.array(x),np.array(y)


def gbrt_training(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.5)
    t = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1)
    t.fit(x_train,y_train)
    p = t.predict(x_test)
    p = map(int,p)
    print ((p-y_test)**2).mean()
    p = t.predict(x)
    p = map(int,p)
    print ((p-y)**2).mean()
    print p[:10]
    print y[:10]

from sklearn.linear_model import LinearRegression
def lr_training(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.5)
    t = LinearRegression()
    t.fit(x_train,y_train)
    p = t.predict(x_test)
    p = map(int,p)
    print ((p-y_test)**2).mean()
    p = t.predict(x)
    p = map(int,p)
    print ((p-y)**2).mean()
    print p[:10]
    print y[:10]


x,y = load_data(SRC_FILE)
gbrt_training(x,y)
lr_training(x,y)