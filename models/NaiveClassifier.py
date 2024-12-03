import tensorflow as tf
from tensorflow.python.keras.models import Model

import tensorflow as tf

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense, MaxPool1D, LeakyReLU, ReLU, Dropout
from keras.src.layers.normalization.batch_normalization import BatchNormalization

from keras.src.layers.normalization.group_normalization import GroupNormalization
from tensorflow.python.keras.optimizers import rmsprop_v2 as RMSprop
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy

from tensorflow.python.keras import Sequential


def make_naive_discriminator(seq_shape:tuple, n_classes:int)-> Model:

    _input = Input(seq_shape)

    x = Conv1D(8, 3, 1, padding='same')(_input) 
    x = BatchNormalization()(x)
    x = MaxPool1D()(x)
    x = ReLU()(x)
    #

    x = Conv1D(16, 3, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D()(x)
    x = ReLU()(x)

    x = Flatten()(x)
    x = Dense(32)(x)
    x = ReLU()(x)
    
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(_input, x)
    
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    
    return model

