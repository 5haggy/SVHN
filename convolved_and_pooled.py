from __future__ import print_function

import pickle
import math
import scipy
from scipy.io import loadmat as load

import mlp
from preprocess import Preprocess

kmeans = pickle.load(open('kmeans.pickle', 'rb'))

x_train = pickle.load(open('xtrain.pickle', 'rb'))
y_train = pickle.load(open('ytrain.pickle', 'rb'))
x_test = pickle.load(open('xtest.pickle', 'rb'))
y_test = pickle.load(open('ytest.pickle', 'rb'))
(x_valid, y_valid) = (x_train[math.floor(0.8*len(x_train)):,:], y_train[math.floor(0.8*len(y_train)):,:])
(x_train, y_train) = (x_train[:math.floor(0.8*len(x_train)),:], y_train[:math.floor(0.8*len(y_train)),:])

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

(name, model) = mlp.create_mlp('relu', 512, 256)

mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)