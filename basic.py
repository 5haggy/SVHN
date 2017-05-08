from __future__ import print_function
from functools import reduce

import math
import scipy
from scipy.io import loadmat as load

import mlp
from preprocess import Preprocess

(x_train, y_train) = Preprocess(load('train_32x32.mat')).rgb2gray().normalize().flatten().get()
(x_test, y_test) = Preprocess(load('test_32x32.mat')).rgb2gray().normalize().flatten().get()

(x_valid, y_valid) = (x_train[math.floor(0.8*len(x_train)):,:], y_train[math.floor(0.8*len(y_train)):,:])
(x_train, y_train) = (x_train[:math.floor(0.8*len(x_train)),:], y_train[:math.floor(0.8*len(y_train)),:])

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

networksParams = [(activationFn, numOfNeurons, 1024, dropOut)
    for dropOut in [0, 0.25, 0.5, 0.75] 
    for numOfNeurons in [256, 512, 1024] 
    for activationFn in ['tanh', 'relu', 'sigmoid']]

models = [mlp.create_mlp(*params) for params in networksParams]

for model in models:
    mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)

scores = [mlp.test_mlp(model, x_test, y_test) for model in models]

highestAccModel = reduce(lambda a,b: a if a[1] > b[1] else b, scores)

print('Highest accuracy: ', highestAccModel[1])