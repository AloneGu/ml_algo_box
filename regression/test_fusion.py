import numpy as np
from regression_cls import MyRegression
import matplotlib.pyplot as plt

def get_x_y_data(file_path):
    x_data = []
    y_data = []
    f = open(file_path)
    for r in f.readlines():
        try:
            #print r
            r = r.strip().split(',')
            #print r
            #tmp_x = [int(r[-2]),int(r[-1]),int(r[3])]
            if int(r[-2])>15:
                continue
            tmp_x = [int(r[-2]),int(r[-1])]
            tmp_y = int(r[-3])
            x_data.append(tmp_x)
            y_data.append(tmp_y)
        except:
            pass
    f.close()
    return x_data,y_data


if __name__ == '__main__':
    import sys
    if len(sys.argv)!=2:
        print 'usage python test_fusion.py ../data/xx.txt'
    x, y = get_x_y_data(sys.argv[1])

    regression_worker = MyRegression(x, y)
    adjust = regression_worker.ols_linear_reg()
    regression_worker.bayes_ridge_reg()
    regression_worker.linear_ridge_reg()

    v = [xx[0] for xx in x]
    w = [vx[1] for vx in x]
    line1, = plt.plot(y,color='r',label='truth')
    line2, = plt.plot(w,color='b',label='wifi')
    plt.plot(v,color='g',label='video')
    plt.plot(adjust,color='k',label='fusion result')
    plt.title('fusion video and wifi')
    plt.ylabel('occupancy of location id 120 ')


    # Create another legend for the second line.
    plt.legend()
    plt.show()





