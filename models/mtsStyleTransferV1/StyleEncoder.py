import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Flatten, Dense
from keras.src.layers.normalization.group_normalization import GroupNormalization

from tensorflow.python.keras.models import Model



def make_style_encoder(seq_length:int, n_feat:int, vector_output_shape:int)  -> Model:

    _input = Input((seq_length, n_feat))

    x = Conv1D(16, 5, 2, padding='same')(_input) 
    x = GroupNormalization(groups=-1)(x)
    x = LeakyReLU()(x)

# ###

    x = Conv1D(32, 5, 2, padding='same')(x)
    x = GroupNormalization(groups=-1)(x)
    x = LeakyReLU()(x)

# ###

    x = Conv1D(64, 5, 2, padding='same')(x)
    x = GroupNormalization(groups=-1)(x)
    x = LeakyReLU()(x)
    
# ###

    x = Flatten()(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dense(vector_output_shape)(x)
    
    x = tf.clip_by_norm(x, 1., -1)

    model = Model(_input, x)
    
    return model