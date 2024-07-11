import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf


def local_discriminator_part(_input, n_classes:int):

    x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    stage1_dropouted = tf.keras.layers.Dropout(0.25)(x)

    _output = tf.keras.layers.Dense(1, activation="sigmoid")(stage1_dropouted)
    # _class_output = layers.Dense(n_classes, activation="sigmoid")(stage1_dropouted)

    return _output


def create_local_discriminator(n_signals:int, sequence_length:int, n_styles:int)  -> tf.keras.Model:
    sig_inputs = tf.keras.Input((sequence_length, n_signals))
    splited_inputs = tf.split(sig_inputs, n_signals, axis=-1)

    crit_outputs = []

    for sig_input in splited_inputs:
        crit_output = local_discriminator_part(sig_input, n_styles)
        crit_outputs.append(crit_output)

    crit_outputs = tf.keras.layers.concatenate(crit_outputs, axis=-1, name="crit_output")

    return tf.keras.Model(sig_inputs, crit_outputs)