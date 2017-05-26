from __future__ import print_function

import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.io import loadmat as load
from preprocess import Preprocess

print('Loading data')
(x,y) = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().get()
ims = np.array([x[i,k:k+8,j:j+8] for k in [0,8,16,24] for j in [0,8,16,24] for i in range(0, np.shape(x)[0])])
ims = ims.reshape(np.shape(ims)[0], 64)
print('KMeans')
kmeans = MiniBatchKMeans(n_clusters=500, init='k-means++',verbose=0).fit(ims)
pickle.dump(kmeans, open('kmeans.pickle', 'wb'))
