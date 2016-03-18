import numpy as np
from regression_cls import MyRegression

def get_x_y_data(file_path):
    x_data = []
    y_data = []
    f = open(file_path)
    for r in f.readlines():
        try:
            #print r
            r = r.strip().split(',')
            print r
            tmp_x = [int(r[-1]),int(r[-2])]
            tmp_y = int(r[-3])
            x_data.append(tmp_x)
            y_data.append(tmp_y)
        except:
            pass
    f.close()
    return x_data,y_data


if __name__ == '__main__':
    import sys
    x, y = get_x_y_data(sys.argv[1])

    regression_worker = MyRegression(x, y)
    regression_worker.ols_linear_reg()
    regression_worker.bayes_ridge_reg()
    regression_worker.linear_ridge_reg()


