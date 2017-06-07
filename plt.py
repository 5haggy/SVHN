from __future__ import print_function

import math
import scipy
from scipy.io import loadmat as load
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

from preprocess import Preprocess

ims = load('train_32x32.mat')['X']
ims = np.array([ims[:,:,:,i] for i in range(0, np.shape(ims)[3])])
gray = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().get()[0]
scaled = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().scale2().get()[0]
binary = Preprocess(load('train_32x32.mat')).rgb2gray().uint8().binarize().get()[0]
binary2 = Preprocess(load('train_32x32.mat')).rgb2gray().scale2().uint8().binarize().get()[0]

n=10
plt.figure(figsize=(50, 8))
for i in range(n):
    j = randrange(0, 100) + i
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(ims[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(gray[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(5, n, i + 1 + n + n)
    plt.imshow(scaled[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(5, n, i + 1 + n + n+n)
    plt.imshow(binary[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(5, n, i + 1 + n + n+n+n)
    plt.imshow(binary2[j])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()