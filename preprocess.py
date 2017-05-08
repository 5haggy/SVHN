from __future__ import print_function

import numpy as np

class Preprocess():
    def __init__(self, data):
        def get_category(num):
            y = np.zeros((1, 10))
            y[:, num - 1] = 1
            return y

        self.x = data['X']
        self.y = np.array([get_category(num[0]) for num in data['y']]).reshape(len(data['y']), 10)

    def rgb2gray(self):
        def it(im):
            return np.dot(im, [0.299, 0.587, 0.114])

        self.x = np.array([it(self.x[:,:,:,i]) for i in range(0, np.shape(self.x)[3])])
        return self

    def normalize(self):
        self.x /= 255
        return self


    def scale(self):
        def it(x):
            return -x*(x-31)/250

        scaleVector = np.array([it(x) for x in range(0,32)])
        self.x = np.array([self.x[i,:,:] * scaleVector for i in range(0, np.shape(self.x)[0])])
        return self

    def flatten(self):
        self.x = self.x.reshape(np.shape(self.x)[0], 1024)
        return self

    def get(self):
        return (self.x, self.y)