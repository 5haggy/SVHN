from __future__ import print_function

import math
import scipy
import pickle

import mlp

(x_train, y_train) = (pickle.load(open('xtrain_sift.pickle', 'rb')), pickle.load(open('ytrain_sift.pickle', 'rb')))
(x_test, y_test) = (pickle.load(open('xtest_sift.pickle', 'rb')), pickle.load(open('ytest_sift.pickle', 'rb')))

(x_valid, y_valid) = (x_train[math.floor(0.8*len(x_train)):,:], y_train[math.floor(0.8*len(y_train)):,:])
(x_train, y_train) = (x_train[:math.floor(0.8*len(x_train)),:], y_train[:math.floor(0.8*len(y_train)),:])

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

(name, model) = mlp.create_mlp(100, ('relu', 512, 0))

mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)

(name, model) = mlp.create_mlp(100, ('relu', 256, 0))

mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)

(name, model) = mlp.create_mlp(100, ('relu', 128, 0))

mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)