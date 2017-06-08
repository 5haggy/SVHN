from __future__ import print_function
from functools import reduce

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import regularizers

def create_mlp(inputs,l2, *layers):
    name = ' '.join([str(val) for layer in layers for val in layer])

    print('------------------')
    print(name)
    print('------------------')

    model = Sequential()

    model.add(Dense(layers[0][1], activation=layers[0][0], input_shape=(inputs,), 
                                  activity_regularizer=regularizers.l2(l2)))
    if layers[0][2] != 0:
        model.add(Dropout(layers[0][2]))

    for layer in layers[1:]:
        model.add(Dense(layer[1], activation=layer[0], activity_regularizer=regularizers.l2(l2)))
        if layer[2] != 0:
            model.add(Dropout(layer[2]))

    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  #optimizer=SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True),
                  #optimizer='rmsprop',
                  optimizer='adagrad',
                  metrics=['accuracy'])

    return (name, model)

def train_mlp(model, numOfEpochs, x_train, y_train, x_valid, y_valid):
    csv_logger = CSVLogger('training {0}.csv'.format(name))
    history =model.fit(x_train, y_train,
                batch_size=128,
                epochs=numOfEpochs,
                verbose=1,
                callbacks=[csv_logger],
                validation_data=(x_valid, y_valid)                )
    print(history.history.keys())
    model.save('{0}.h5'.format(name))
    
# summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{0} accurancy.png'.format(name), bbox_inches='tight')
    plt.show()
# summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{0} loss.png'.format(name), bbox_inches='tight')
    plt.show()
    return 

def test_mlp(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score
