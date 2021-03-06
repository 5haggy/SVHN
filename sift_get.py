from __future__ import print_function

import pickle
from scipy.io import loadmat as load

from preprocess import Preprocess

kmeans = pickle.load(open('sift/sift_kmeans.pickle', 'rb'))

(x_train, y_train) = Preprocess(load('train_32x32.mat')).rgb2gray().uint8().sift(kmeans).get()
pickle.dump(x_train, open('sift/xtrain_sift.pickle', 'wb'))
pickle.dump(y_train, open('ytrain_sift.pickle', 'wb'))

(x_test, y_test) = Preprocess(load('test_32x32.mat')).rgb2gray().uint8().sift(kmeans).get()
pickle.dump(x_test, open('sift/xtest_sift.pickle', 'wb'))
pickle.dump(y_test, open('sift/ytest_sift.pickle', 'wb'))