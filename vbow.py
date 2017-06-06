from __future__ import print_function

import math
import pickle

import mlp
from sklearn import svm

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
import matplotlib.pyplot as plt

import scipy
from scipy.io import loadmat as load
from preprocess import Preprocess

(x_train, y_train) = (pickle.load(open('xtrain_sift.pickle', 'rb')), pickle.load(open('ytrain_sift.pickle', 'rb')))
(x_test, y_test) = (pickle.load(open('xtest_sift.pickle', 'rb')), pickle.load(open('ytest_sift.pickle', 'rb')))

(x_valid, y_valid) = (x_train[math.floor(0.8*len(x_train)):,:], y_train[math.floor(0.8*len(y_train)):,:])
(x_train, y_train) = (x_train[:math.floor(0.8*len(x_train)),:], y_train[:math.floor(0.8*len(y_train)),:])

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

input_img = Input(shape=(500,))
sftmx = Dense(10, activation='softmax')(input_img)

model = Model(input_img, sftmx)

model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  #optimizer='rmsprop',
                  metrics=['accuracy'])

model.fit(x_train, y_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_valid, y_valid))

score = model.evaluate(x_test, y_test, verbose=0)

y_train = np.where(y_train==1)[1]
y_test = np.where(y_test==1)[1]

svc = svm.LinearSVC()
svc.fit(x_train, y_train)

score = svc.score(x_test, y_test)

print(score)

"""
(name, model) = mlp.create_mlp(500, ('relu', 512, 0))

mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)

(name, model) = mlp.create_mlp(500, ('relu', 256, 0))

mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)

(name, model) = mlp.create_mlp(500, ('relu', 1024, 0))

mlp.train_mlp(model, 20, x_train, y_train, x_valid, y_valid)
mlp.test_mlp(model, x_test, y_test)
"""