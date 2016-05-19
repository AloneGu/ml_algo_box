import numpy as np
from regression_cls import MyRegression

v = [0, 8, 13, 13, 9, 1, 0, 0, 11, 6, 5, 0, 13, 5, 15, 9, 19, 14, 20, 12, 9, 1, 8, 10, 13, 1, 0, 6, 8, 19, \
     6, 3, 12, 11, 9, 11, 6, 11, 2, 7, 3, 0, 2, 8, 12, 18, 11, 1, 7, 0, 6, 12, 6, 5, 0, 25, 16, 0, 26, 1, 24, \
     20, 28, 8, 24, 20, 15, 0, 0, 24, 15, 12, 19, 18, 13, 20, 16, 16, 20, 22, 22, 0, 17, 0, 18, 19, 12, 22, 20, 20, 12,
     0, 8, 0, 0, 18]

w = [6, 21, 21, 23, 18, 0, 0, 5, 10, 14, 19, 0, 20, 13, 18, 8, 19, 24, 20, 23, 19, 19, 18, 23, 21, 14, 0, 17, 19, \
     18, 17, 22, 19, 15, 13, 22, 16, 16, 24, 17, 13, 0, 0, 17, 25, 18, 19, 22, 21, 0, 22, 16, 17, 21, 0, 20, 14, 8, \
     25, 26, 19, 20, 21, 10, 23, 14, 19, 7, 7, 24, 26, 18, 28, 14, 10, 17, 21, 20, 25, 22, 24, 0, 21, 0, 23, 12, 26, 18,
     17, 20, 17, 19, 24, 0, 21, 22]

a = np.random.randint(2, size=96)
a = map(lambda x: 1 if x == 1 else -1, a)

g = np.random.randint(-10, 10, size=96) + v


def get_x_y_data(x_lists, y):
    x_data = []
    y_data = []
    for v, w, a in x_lists:
        x_data.append([v, w, a])
    for g in y:
        y_data.append(g)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


if __name__ == '__main__':
    x, y = get_x_y_data(zip(v, w, a), g)
    regression_worker = MyRegression(x, y)
    regression_worker.ols_linear_reg()
    regression_worker.bayes_ridge_reg()
    regression_worker.linear_ridge_reg()
    print regression_worker.lr.predict([8,21,1])
