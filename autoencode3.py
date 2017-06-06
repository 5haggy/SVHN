from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
import matplotlib.pyplot as plt

import pickle
import scipy
from scipy.io import loadmat as load
from preprocess import Preprocess

# this is the size of our encoded representations
encoding_dim = 64

# this is our input placeholder
input_img = Input(shape=(64,))
# "encoded" is the encoded representation of the input
sftmx = Dense(10, activation='softmax')(input_img)
# "decoded" is the lossy reconstruction of the input

# this model maps an input to its reconstruction
model = Model(input_img, sftmx)

model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  #optimizer='rmsprop',
                  metrics=['accuracy'])

x_train = pickle.load(open('f2train.pickle', 'rb'))
x_test = pickle.load(open('f2test.pickle', 'rb'))

y_train = Preprocess(load('train_32x32.mat')).get()[1]
y_test = Preprocess(load('test_32x32.mat')).get()[1]

model.fit(x_train, y_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

