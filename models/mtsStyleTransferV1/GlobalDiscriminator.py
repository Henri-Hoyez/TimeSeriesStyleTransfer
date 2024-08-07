import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

def make_global_discriminator(seq_length:int, n_signals:int, n_classes:int)  -> tf.keras.Model:
    _input = tf.keras.Input((seq_length, n_signals))

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same')(_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    crit_hidden_layer = tf.keras.layers.Dense(10)(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(crit_hidden_layer)
    _class_output = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)

    model = tf.keras.Model(_input, [_output, _class_output], name="global_discriminator")

    return model