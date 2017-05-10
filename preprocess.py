from __future__ import print_function

import cv2
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

    def sift(self):
        def it(im):
            sift = cv2.xfeatures2d.SIFT_create()
            return sift.detectAndCompute(im, None)

        self.sift_kp = np.array([it(self.x[i,:,:]) for i in range(0, np.shape(self.x)[0])])
        return self

    def convolve_pool(self, kmeans):
        def pool(im):
            def poolIt(k):
                sum = np.zeros((64,))
                for i in range(0,6):
                    for j in range(0,6):
                        sum += im[k+i,k+j,:]
                return sum/36

            return np.array([poolIt(k) for k in [0,6,12,18]]).reshape(256,)

        def it(im):
            return pool(np.array([[kmeans.cluster_centers_[kmeans.predict(im[i:i+8,j:j+8].reshape(1,64))[0]] for i in range(0,24)] for j in range(0,24)]))

        self.x = np.array([it(self.x[i,:,:]) for i in range(0, np.shape(self.x)[0])])
        return self

    def scale(self):
        def it(x):
            return -x*(x-31)/250

        scaleVector = np.array([it(x) for x in range(0,32)])
        self.x = np.array([self.x[i,:,:] * scaleVector for i in range(0, np.shape(self.x)[0])])
        return self

    def scale2(self):
        import math
        def it(x):
            return 17.5/(7*((2*math.pi)**(1/2)))*math.exp(-(x-16)**2/98)

        scaleVector = np.array([it(x) for x in range(0,32)])
        self.x = np.array([self.x[i,:,:] * scaleVector for i in range(0, np.shape(self.x)[0])])
        return self

    def uint8(self):
        self.x = self.x.astype('uint8')
        return self

    def flatten(self):
        self.x = self.x.reshape(np.shape(self.x)[0], 1024)
        return self

    def get(self):
        return (self.x, self.y)