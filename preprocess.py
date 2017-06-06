from __future__ import print_function

import cv2
import numpy as np
import keras
from scipy.spatial.distance import euclidean
from functools import reduce
from skimage.feature import hog

class Preprocess():
    def __init__(self, data):
        self.x = data['X']
        self.y = data['y']
        self.y = self.y.reshape(len(self.y),)
        self.y[self.y == 10] = 0
        self.y = keras.utils.to_categorical(self.y, 10)
        #self.x = np.array([self.x[:,:,:,i] for i in range(0, np.shape(self.x)[3])])

    def rgb2gray(self):
        def it(im):
            return np.dot(im, [0.299, 0.587, 0.114])

        self.x = np.array([it(self.x[:,:,:,i]) for i in range(0, np.shape(self.x)[3])])
        return self

    def normalize(self):
        self.x /= 255
        return self

    def standarize(self):
        def it(im):
            flat = im.reshape(1, 1024)
            (mean, sd) = (np.mean(flat), np.std(flat))
            return (im - mean)/sd

        self.x = np.array([it(x) for x in self.x])
        return self

    def binarize(self):
        self.x = np.array([cv2.threshold(x,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for x in self.x])
        return self

    def hog(self):
        self.x = np.array([hog(x, orientations=9, pixels_per_cell=(6, 6),block_norm="L2") for x in self.x])
        return self

    def sift(self, kmeans):
        sift = cv2.xfeatures2d.SIFT_create()

        def it(im):
            points = np.array(sift.detectAndCompute(im, None)[1])
            bov = np.zeros((len(kmeans.cluster_centers_),))
            if np.shape(points) == ():
                points = np.zeros((0,128))
            for point in points:
                bov[kmeans.predict(point)[0]] += 1
            return bov

        self.x = np.array([it(im) for im in self.x])
        return self

    def convolve_pool(self, kmeans):
        def pool(im):
            def poolIt(k,l):
                sum = np.zeros((500,))
                for i in range(0,6):
                    for j in range(0,6):
                        sum += im[k+i,l+j,:]
                return sum/36

            return np.array([poolIt(k,l) for k in [0,6] for l in [0,6]]).reshape(2000,)

        def convolve(im):
            distances = [euclidean(im, cc) for cc in kmeans.cluster_centers_]
            mean = np.mean(distances)
            return np.array([max(0, mean - distance) for distance in distances])
        def it(im,iters):
            print(iters)
            return pool(np.array([[convolve(im[i:i+8,j:j+8].reshape(64,)) for i in range(0,24,2)] for j in range(0,24,2)]))

        self.x = np.array([it(self.x[i,:,:], i) for i in range(0, np.shape(self.x)[0])])
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