import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate
from tensorflow.python.keras.models import Model


def make_global_discriminator(seq_length:int, n_signals:int, n_classes:int):

    inputs = [Input((seq_length, 1)) for _ in range(n_signals)]

    _input = Concatenate(-1)(inputs)

    x = Conv1D(32, 5, 2, padding='same')(_input) # 64
    x = LeakyReLU()(x)
    
    x = Conv1D(64, 5, 2, padding='same')(_input) # 64
    x = LeakyReLU()(x)

    x = Conv1D(128, 5, 2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv1D(256, 5, 2, padding='same')(x)
    x = LeakyReLU()(x)

    flatened = Flatten()(x)
    flatened = Dropout(0.0)(flatened)
    
    crit_hidden_layer = Dense(10)(flatened)
    _output = Dense(1, activation="sigmoid")(crit_hidden_layer)

    class_hidden = Dense(150)(flatened)
    class_hidden = LeakyReLU()(class_hidden)
    
    class_hidden = Dense(50)(class_hidden)
    class_hidden = LeakyReLU()(class_hidden)
    
    # class_hidden = Dense(10, activation='relu')(class_hidden)
    _class_output = Dense(n_classes, activation="softmax")(class_hidden)

    model = Model(inputs, [_output, _class_output], name="global_discriminator")
    
    model.summary()
    # exit()

    return model

