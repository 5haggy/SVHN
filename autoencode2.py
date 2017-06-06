from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

import pickle
import scipy
from scipy.io import loadmat as load
from preprocess import Preprocess

# this is the size of our encoded representations
encoding_dim = 64

# this is our input placeholder
input_img = Input(shape=(144,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(144, activation='sigmoid')(encoded)

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

autoencoder.compile(optimizer='adadelta', loss='mse')

x_train = pickle.load(open('f1train.pickle', 'rb'))
x_test = pickle.load(open('f1test.pickle', 'rb'))

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs_train = encoder.predict(x_train)
encoded_imgs_test = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs_train)

pickle.dump(encoded_imgs_train, open('f2train.pickle', 'wb'))
pickle.dump(encoded_imgs_test, open('f2test.pickle', 'wb'))

plt.imshow(decoded_imgs[1].reshape(32,32))
plt.show()