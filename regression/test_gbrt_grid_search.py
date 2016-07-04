import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import csv
SRC_FILE = '../data/113_2016-04-13_occupancy_minutes.csv'

def load_data(f):
    csv_r = csv.reader(open(f))
    csv_r.next() # jump header
    x,y=[],[]
    l_t = []
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
        l_t.append([r[0],tmp_t])
    return l_t,np.array(x),np.array(y)


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

def gbrt_training_save_res(x,y):
    t = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1)
    t.fit(x,y)
    p = t.predict(x)
    f = lambda x:max(0,int(x))
    p = map(f,p)
    return p


l_t,x,y = load_data(SRC_FILE)
gbrt_training(x,y)
#lr_training(x,y)
# save result to csv
predicted_res = gbrt_training_save_res(x,y)
x_len = len(x)
fn = '../data/'+l_t[0][0]+'_res.csv'
with open(fn,'w') as fout:
    csv_w = csv.writer(fout)
    csv_w.writerow(['location','time','hour','minutes','video occ minutes','wifi occ minutes','video occupancy','groundtruth occ minutes','fusion occ minutes'])
    for i in range(x_len):
        print l_t[i]
        print x[i]
        print y[i]
        print predicted_res[i]
        csv_w.writerow(l_t[i]+list(x[i])+[y[i]]+[predicted_res[i]])
