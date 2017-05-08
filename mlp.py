from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

def create_mlp(activationFn, numOfNeurons, numOfInputs, dropOut = 0):
    name = activationFn + ' ' + str(numOfNeurons) + ' ' + str(dropOut)

    print('------------------')
    print(name)
    print('------------------')

    model = Sequential()

    model.add(Dense(numOfNeurons, activation=activationFn, input_shape=(numOfInputs,)))

    if dropOut != 0:
        model.add(Dropout(dropOut))

    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    return (name, model)

def train_mlp(model, numOfEpochs, x_train, y_train, x_valid, y_valid):
    model.fit(x_train, y_train,
                batch_size=128,
                epochs=numOfEpochs,
                verbose=0,
                validation_data=(x_valid, y_valid))

def test_mlp(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score