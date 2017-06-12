from __future__ import print_function

import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.io import loadmat as load
from preprocess import Preprocess

kmeans = MiniBatchKMeans(n_clusters=500, init='k-means++',verbose=0).fit(pickle.load(open('sift/sift.pickle', 'rb')))
pickle.dump(kmeans, open('sift/sift_kmeans.pickle', 'wb'))