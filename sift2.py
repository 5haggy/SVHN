from __future__ import print_function

import pickle
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.io import loadmat as load
from preprocess import Preprocess

print('Loading data')
(x,y) = Preprocess(load('train_32x32.mat')).rgb2gray().uint8().get()
sift = cv2.xfeatures2d.SIFT_create()
points = np.array(sift.detectAndCompute(x[0,:,:], None)[1])
print(np.shape(points))
for i in range(1, np.shape(x)[0]):
    next = np.array(sift.detectAndCompute(x[i,:,:], None)[1])
    if np.shape(next) == ():
        next = np.zeros((0,128))
    points = np.concatenate((points, next), axis=0)
#points = np.array([sift.detectAndCompute(x[i,:,:], None)[1] for i in range(0, np.shape(x)[0])])
#print('KMeans')
#kmeans = MiniBatchKMeans(n_clusters=500, init='k-means++',verbose=0).fit(points)
pickle.dump(points, open('sift.pickle', 'wb'))