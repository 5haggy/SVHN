from __future__ import print_function

import math
import scipy
from scipy.io import loadmat as load
import numpy as np
import matplotlib.pyplot as plt

import mlp
from preprocess import Preprocess

(x_train, y_train) = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().scale().flatten().get()

(x_test, y_test) = Preprocess(load('test_32x32.mat')).rgb2gray().standarize().scale().flatten().get()

(x_valid, y_valid) = (x_train[math.floor(0.8*len(x_train)):,:], y_train[math.floor(0.8*len(y_train)):,:])
(x_train, y_train) = (x_train[:math.floor(0.8*len(x_train)),:], y_train[:math.floor(0.8*len(y_train)),:])

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

(name, model) = mlp.create_mlp(1024, 0, ('relu', 512, 0))

mlp.train_mlp(model, 50, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)

w = np.squeeze(model.layers[0].get_weights())
hw = w[0].T
plot_weights(hw,32)
