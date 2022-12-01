from keras import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as k


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        if k.image_data_format() == 'channels_first':
            inputShape = (depth,height,width)
        else:
            inputShape = (height,width,depth)
        model.add(Conv2D(20,(5,5),padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),(2,2)))
        model.add(Conv2D(50,(5,5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),(2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model