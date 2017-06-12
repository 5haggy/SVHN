from __future__ import print_function

import math
import scipy
from scipy.io import loadmat as load
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import cv2

from preprocess import Preprocess

x = Preprocess(load('train_32x32.mat')).rgb2gray().uint8().get()[0]
scaled = Preprocess(load('train_32x32.mat')).rgb2gray().scale2().uint8().get()[0]
sift = cv2.xfeatures2d.SIFT_create()

n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    j = randrange(0, 100) + i
    ax = plt.subplot(2, n, i + 1)
    kp = sift.detectAndCompute(x[j,:,:], None)[0]
    plt.imshow(cv2.drawKeypoints(x[j,:,:], kp, x[j,:,:]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + 1 + n)
    kp2 = sift.detectAndCompute(scaled[j,:,:], None)[0]
    plt.imshow(cv2.drawKeypoints(scaled[j,:,:], kp2, scaled[j,:,:]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()