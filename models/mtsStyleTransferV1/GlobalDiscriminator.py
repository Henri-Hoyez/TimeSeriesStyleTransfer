import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

def make_global_discriminator(seq_length:int, n_signals:int, n_classes:int):

    inputs = [tf.keras.Input((seq_length, 1)) for _ in range(n_signals)]

    _input = tf.keras.layers.Concatenate(-1)(inputs)

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same')(_input)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)


    flatened = tf.keras.layers.Flatten()(_input)
    crit_hidden_layer = tf.keras.layers.Dense(10)(flatened)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(crit_hidden_layer)


    class_hidden = tf.keras.layers.Dense(50, activation='relu')(flatened)
    # class_hidden = tf.keras.layers.Dense(50, activation='relu')(class_hidden)
    # class_hidden = tf.keras.layers.Dense(10, activation='relu')(class_hidden)
    _class_output = tf.keras.layers.Dense(n_classes, activation="softmax")(class_hidden)

    model = tf.keras.Model(inputs, [_output, _class_output], name="global_discriminator")

    return model

