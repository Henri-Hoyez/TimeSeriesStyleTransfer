import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa



def local_discriminator_part(_input, n_classes:int):

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv1D(32, 5, 2, padding='same'))(_input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv1D(32, 5, 2, padding='same'))(x) # 64
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    stage1_dropouted = tf.keras.layers.Dropout(0.0)(x)

    _output = tf.keras.layers.Dense(1, activation="sigmoid")(stage1_dropouted)

    return _output

def create_local_discriminator(n_signals:int, sequence_length:int, n_styles:int):

    inputs = [tf.keras.Input((sequence_length, 1)) for _ in range(n_signals)]
    crit_outputs = []

    for sig_input in inputs:
        crit_output = local_discriminator_part(sig_input, n_styles)
        crit_outputs.append(crit_output)

    return tf.keras.Model(inputs, crit_outputs)