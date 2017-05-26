from __future__ import print_function

import math
import scipy
import pickle
import numpy as np
from sklearn import svm

x_train = pickle.load(open('xtrain.pickle', 'rb'))
y_train = pickle.load(open('ytrain.pickle', 'rb'))
x_test = pickle.load(open('xtest.pickle', 'rb'))
y_test = pickle.load(open('ytest.pickle', 'rb'))
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = np.where(y_train==1)[1]
y_test = np.where(y_test==1)[1]

svc = svm.LinearSVC()
svc.fit(x_train, y_train)

score = svc.score(x_test, y_test)

print(score)