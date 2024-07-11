import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

def make_content_encoder(seq_length:int, n_feat:int, feat_wiener:int) -> tf.keras.Model:
    init = tf.keras.initializers.RandomNormal(seed=42)

    _input = tf.keras.Input((seq_length, n_feat))

    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same', kernel_initializer=init)(_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
        
    # x = tf.keras.layers.Conv1D(64, 5, 1, padding='same', kernel_initializer=init)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv1D(64, 5, 2, padding='same', kernel_initializer=init)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv1D(feat_wiener, 5, 1, padding='same', kernel_initializer=init, activation="linear")(x)

    model = tf.keras.Model(_input, x)
    return model