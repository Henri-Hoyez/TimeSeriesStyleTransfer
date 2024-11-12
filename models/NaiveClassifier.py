import tensorflow as tf
from tensorflow.python.keras.models import Model

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense, MaxPool1D
from keras.src.layers.normalization.group_normalization import GroupNormalization
from tensorflow.python.keras.optimizers import rmsprop_v2 as RMSprop
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy

from tensorflow.python.keras import Sequential


def make_naive_discriminator(seq_shape:tuple, n_classes:int)-> Model:
    # initializer = tf.keras.initializers.GlorotNormal()

    model = Sequential()


    model.add(Input(shape=seq_shape))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding="same"))
    # model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(filters=256, kernel_size=3, padding="same", activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    # model.add(LeakyReLU())
    
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    # model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    # model.add(LeakyReLU())


    model.add(Flatten())

    # model.add(Dense(units=100, activation='sigmoid'))
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=50, activation="relu"))
    # model.add(LeakyReLU())
    # model.add(Activation("relu"))

    
    # model.add(Dropout(0.5))
    model.add(Dense(units=n_classes, activation="softmax"))
    # model.add(Dense(units=n_classes))

    model.compile(
        optimizer=RMSprop.RMSprop(),
        loss=SparseCategoricalCrossentropy(from_logits=False),
        metrics=SparseCategoricalAccuracy()
    )
    
    # print(model.summary())

    return model

