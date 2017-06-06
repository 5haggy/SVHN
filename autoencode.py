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

# this is the size of our encoded representations
encoding_dim = 144

# this is our input placeholder
input_img = Input(shape=(1024,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(1024, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile('adadelta', loss='binary_crossentropy')

(x_train, y_train) = Preprocess(load('train_32x32.mat')).rgb2gray().normalize().flatten().get()

(x_test, y_test) = Preprocess(load('test_32x32.mat')).rgb2gray().normalize().flatten().get()

noise_factor = 0.05
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

autoencoder.fit(x_train_noisy, x_train,
                epochs=25,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

w = np.squeeze(autoencoder.layers[1].get_weights())
hw = w[0].T
l = autoencoder.layers[1]
pickle.dump(w, open('w1.pickle', 'wb'))
pickle.dump(l, open('l1.pickle', 'wb'))

encoded_imgs_train = encoder.predict(x_train_noisy)
encoded_imgs_test = encoder.predict(x_test_noisy)
decoded_imgs = decoder.predict(encoded_imgs_test)

pickle.dump(encoded_imgs_train, open('f1train.pickle', 'wb'))
pickle.dump(encoded_imgs_test, open('f1test.pickle', 'wb'))
#pickle.dump(autoencoder, open('ae1.pickle', 'wb'))
pickle.dump(encoder, open('e1.pickle', 'wb'))
#pickle.dump(decoder, open('d1.pickle', 'wb'))

n=10
plt.figure(figsize=(30, 8))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(encoded_imgs_test[i].reshape(12, 12))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

n=10
plt.figure(figsize=(30, 8))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(hw[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(hw[i+n].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(hw[n+n+i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#weights = autoencoder.layers[0].W.get_value(borrow=True)

"""
# this is the size of our encoded representations
encoding_dim2 = 64

# this is our input placeholder
input_img2 = Input(shape=(144,))
# "encoded" is the encoded representation of the input
encoded2 = Dense(encoding_dim2, activation='relu')(input_img2)
# "decoded" is the lossy reconstruction of the input
decoded2 = Dense(144, activation='sigmoid')(encoded2)

# this model maps an input to its reconstruction
autoencoder2 = Model(input_img2, decoded2)

# this model maps an input to its encoded representation
encoder2 = Model(input_img2, encoded2)

# create a placeholder for an encoded (32-dimensional) input
encoded_input2 = Input(shape=(encoding_dim2,))
# retrieve the last layer of the autoencoder model
decoder_layer2 = autoencoder2.layers[-1]
# create the decoder model
decoder2 = Model(encoded_input2, decoder_layer2(encoded_input2))

autoencoder2.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder2.fit(encoded_imgs_train, encoded_imgs_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_imgs_test, encoded_imgs_test))

encoded_imgs_train2 = encoder2.predict(encoded_imgs_train)
encoded_imgs_test2 = encoder2.predict(encoded_imgs_test)

pickle.dump(encoded_imgs_train2, open('f2train.pickle', 'wb'))
pickle.dump(encoded_imgs_test2, open('f2test.pickle', 'wb'))

# this is our input placeholder
input_img3 = Input(shape=(256,))
# "encoded" is the encoded representation of the input
sftmx = Dense(10, activation='softmax')(input_img3)
# "decoded" is the lossy reconstruction of the input

# this model maps an input to its reconstruction
model3 = Model(input_img3, sftmx)

model3.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  #optimizer='rmsprop',
                  metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model3.fit(encoded_imgs_train, y_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_imgs_test, y_test))

score = model3.evaluate(encoded_imgs_test, y_test, verbose=0)
print(score)

model4 = Sequential([encoded, sftmx])

model4.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  #optimizer='rmsprop',
                  metrics=['accuracy'])

model4.fit(x_train, y_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))

score = model4.evaluate(x_test, y_test, verbose=0)
print(score)
"""

