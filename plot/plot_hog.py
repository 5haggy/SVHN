from __future__ import print_function

import math
import scipy
from scipy.io import loadmat as load
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import cv2

from preprocess import Preprocess

x = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().hog((8,8)).get()[0]
x2 = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().hog((6,6)).get()[0]
x3 = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().hog((4,4)).get()[0]
n=10
plt.figure(figsize=(50, 8))
for i in range(n):
    j = randrange(0, 1000) + i
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(x[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(x2[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(5, n, i + 1 + n + n)
    plt.imshow(x3[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()