import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge


def get_accuracy(a, b):
    acc_rate = []
    for x, y in zip(a, b):
        if y == 0:
            tmp_err_rate = 0 if x == 0 else 1
        else:
            abs_diff = abs(x - y)
            tmp_err_rate = abs(abs_diff * 1.0 / y)
            tmp_err_rate = min(1.0, tmp_err_rate)
        tmp_acc_rate = 1 - tmp_err_rate
        #print x,y,tmp_acc_rate
        acc_rate.append((tmp_acc_rate))
    return np.average(acc_rate)


class MyRegression(object):
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.v_data = [x[0] for x in x_data]
        print 'old accuracy', get_accuracy(self.v_data, self.y_data)

    def ols_linear_reg(self):
        lr = LinearRegression()
        lr.fit(self.x_data, self.y_data)
        adjusted_result = lr.predict(self.x_data)
        print 'lr params', lr.coef_, lr.intercept_
        print 'lr accuracy', get_accuracy(adjusted_result, self.y_data)
        return map(int, list(adjusted_result))

    def bayes_ridge_reg(self):
        br = BayesianRidge()
        br.fit(self.x_data, self.y_data)
        adjusted_result = br.predict(self.x_data)
        print 'bayes ridge params', br.coef_, br.intercept_
        print 'bayes ridge accuracy', get_accuracy(adjusted_result, self.y_data)
        return map(int, list(adjusted_result))

    def linear_ridge_reg(self):
        rr = Ridge()
        rr.fit(self.x_data, self.y_data)
        adjusted_result = rr.predict(self.x_data)
        print 'ridge params', rr.coef_, rr.intercept_
        print 'ridge accuracy', get_accuracy(adjusted_result, self.y_data)
        return map(int, list(adjusted_result))
