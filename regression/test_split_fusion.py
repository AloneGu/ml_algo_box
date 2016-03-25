import numpy as np
from regression_cls import MyRegression,get_accuracy
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def get_x_y_data(file_path):
    x_data = []
    y_data = []
    f = open(file_path)
    for r in f.readlines():
        try:
            #print r
            r = r.strip().split(',')
            tmp_x = [int(r[-2]),int(r[-1])]
            tmp_y = int(r[-3])
            x_data.append(tmp_x)
            y_data.append(tmp_y)
        except:
            pass
    f.close()
    X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.4)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    import sys
    if len(sys.argv)!=2:
        print 'usage python test_split_fusion.py ../data/xx.txt'
    X_train, X_test, y_train, y_test = get_x_y_data(sys.argv[1])


    regression_worker = MyRegression(X_train, y_train)
    regression_worker.ols_linear_reg()
    regression_worker.bayes_ridge_reg()
    regression_worker.linear_ridge_reg()

    adjust = regression_worker.lr.predict(X_test)

    v = [xx[0] for xx in X_test]
    w = [vx[1] for vx in X_test]
    plt.plot(y_test,color='r',label='truth')
    #plt.plot(w,color='b',label='wifi')
    #plt.plot(v,color='g',label='video')
    plt.plot(adjust,color='k',label='fusion result')
    plt.title('fusion video and wifi')
    plt.ylabel('occupancy of location id 120 ')
    print 'test accuracy',get_accuracy(adjust,y_test)
    video_test = [x[0] for x in X_test]
    print 'train count',len(X_train)
    print 'test count',len(y_test)
    print 'video var',np.var([(x-y) for x,y in zip(video_test,y_train)])
    print 'test var',np.var([(x-y) for x,y in zip(adjust,y_test)])

    # Create another legend for the second line.
    plt.legend()
    plt.show()





