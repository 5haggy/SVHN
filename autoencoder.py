import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import pickle
from keras import regularizers
import numpy as np

import scipy
from scipy.io import loadmat as load
from preprocess import Preprocess
from plot import plot_weights

encoding_dim = 1024

input_img = Input(shape=(1024,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(1024, activation='linear')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile('adadelta', loss='mse')

(x_train, y_train) = Preprocess(load('train_32x32.mat')).rgb2gray().standarize().scale().flatten().get()

(x_test, y_test) = Preprocess(load('test_32x32.mat')).rgb2gray().standarize().scale().flatten().get()

noise_factor = 0.05
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

autoencoder.fit(x_train_noisy, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

encoded_imgs_train = encoder.predict(x_train)
encoded_imgs_test = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs_test)

encoding_dim2 = 256

input_img2 = Input(shape=(encoding_dim,))
encoded2 = Dense(encoding_dim2, activation='relu')(input_img2)
decoded2 = Dense(encoding_dim, activation='linear')(encoded2)

autoencoder2 = Model(input_img2, decoded2)

encoder2 = Model(input_img2, encoded2)

encoded_input2 = Input(shape=(encoding_dim2,))
decoder_layer2 = autoencoder2.layers[-1]
decoder2 = Model(encoded_input2, decoder_layer2(encoded_input2))

autoencoder2.compile(optimizer='adadelta', loss='mse')

autoencoder2.fit(encoded_imgs_train, encoded_imgs_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_imgs_test, encoded_imgs_test))

encoded_imgs_train2 = encoder2.predict(encoded_imgs_train)
encoded_imgs_test2 = encoder2.predict(encoded_imgs_test)

input_sftmx = Input(shape=(encoding_dim2,))
sftmx = Dense(10, activation='softmax')(input_sftmx)
sftmx_model = Model(input_sftmx, sftmx)

sftmx_model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  #optimizer='rmsprop',
                  metrics=['accuracy'])

sftmx_model.fit(encoded_imgs_train2, y_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_imgs_test2, y_test))

score = sftmx_model.evaluate(encoded_imgs_test2, y_test, verbose=0)
print(score)

final = Sequential()
final.add(Dense(encoding_dim, activation='relu', input_shape=(1024,)))
final.add(Dense(encoding_dim2, activation='relu'))
final.add(Dense(10, activation='softmax'))

final.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  #optimizer='rmsprop',
                  metrics=['accuracy'])

final.layers[0].set_weights(autoencoder.layers[1].get_weights())
final.layers[1].set_weights(autoencoder2.layers[1].get_weights())
final.layers[-1].set_weights(sftmx_model.layers[-1].get_weights())

final.fit(x_train, y_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))

score = final.evaluate(x_test, y_test, verbose=0)
print(score)

n=10
plt.figure(figsize=(30, 8))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(encoded_imgs_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()