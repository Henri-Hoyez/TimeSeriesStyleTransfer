import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import keras
from tensorflow.python.keras.regularizers import l2

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten
from tensorflow.python.keras.models import Model


from tensorflow.keras.layers import BatchNormalization, SpectralNormalization



def local_discriminator_part(_input, n_classes:int):

    x = Conv1D(32, 5, 2, padding='same')(_input)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 5, 2, padding='same')(x) # 64
    
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    stage1_dropouted = Dropout(0.0)(x)

    _output = Dense(1, activation="sigmoid")(stage1_dropouted)

    return _output

def create_local_discriminator(n_signals:int, sequence_length:int, n_styles:int):

    inputs = [Input((sequence_length, 1)) for _ in range(n_signals)]
    crit_outputs = []

    for sig_input in inputs:
        crit_output = local_discriminator_part(sig_input, n_styles)
        crit_outputs.append(crit_output)

    return Model(inputs, crit_outputs)