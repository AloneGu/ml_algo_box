#!/usr/bin/env python
# encoding: utf-8


"""
@author: Jackling Gu
@file: my_knn.py
@time: 17-8-7 09:54
"""
import numpy as np

class MyKnn(object):
    def __init__(self):
        self.x_data = None
        self.y_data = None

    def fit(self,x,y):
        """

        :param x: x list
        :param y: y list
        :return:
        """
        self.x_data = np.array(x)
        self.y_data = np.array(y)
        self.data_cnt = len(x)

    def predict(self,x):
        """

        :param x: x list
        :return:  y labels res distance
        error return 1 100
        """
        return [self.process_one_x(tmp_x) for tmp_x in x]

    def process_one_x(self,x):
        # x is  [1,1,xxx]
        # return (label,dis)
        label = 1
        shortest_dis = 100
        curr_x = np.array(x)
        for i in range(self.data_cnt):
            diff_vec = curr_x - self.x_data[i]
            if diff_vec.min() < 0 : # less word , not allowed
                continue
            else:
                distance = diff_vec.sum()
                if distance < shortest_dis: # find a label
                    label = self.y_data[i]
                    shortest_dis = distance
        return (label,shortest_dis)

if __name__ == '__main__':
    t = MyKnn()
    x = [
        [1,1,1,0,0],
        [1,1,0,0,1],
        [0,0,1,1,0]
    ]
    y = [1,2,3]
    test_x = [
        [1,0,1,0,0],
        [1,1,1,0,1],
        [0,0,1,1,0],
        [1,0,1,0,1]
    ]
    t.fit(x,y)
    print(t.predict(x)) # should be [(1,0),(2,0),(3,0)]
    print(t.predict(test_x)) # should be [(1,100),(1,1),(3,0),(1,100)]
