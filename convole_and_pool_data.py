import pickle
from scipy.io import loadmat as load

from preprocess import Preprocess

kmeans = pickle.load(open('kmeans.pickle', 'rb'))

(x_train, y_train) = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().convolve_pool(kmeans).get()
pickle.dump(x_train, open('xtrain.pickle', 'wb'))
pickle.dump(y_train, open('ytrain.pickle', 'wb'))
(x_test, y_test) = Preprocess(load('test_32x32.mat')).rgb2gray().standarize().convolve_pool(kmeans).get()
pickle.dump(x_test, open('xtest.pickle', 'wb'))
pickle.dump(y_test, open('ytest.pickle', 'wb'))