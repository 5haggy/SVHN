import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from preprocess import Preprocess
from scipy.io import loadmat as load

(x_train, y_train) = Preprocess(load('train_32x32.mat')).rgb2gray().normalize().get()
(x_test, y_test) = Preprocess(load('test_32x32.mat')).rgb2gray().normalize().get()

x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

padding = 'valid'

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1), padding=padding))
model.add(Conv2D(16, (3, 3), activation='relu', padding=padding))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding=padding))
model.add(Conv2D(32, (3, 3), activation='relu', padding=padding))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))