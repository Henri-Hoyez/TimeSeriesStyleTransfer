import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf



def make_style_encoder(seq_length:int, n_feat:int, vector_output_shape:int)  -> tf.keras.Model:

    _input = tf.keras.Input((seq_length, n_feat))

    x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(_input) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(32, 5, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # x = tf.keras.layers.Conv1D(16, 5, 1, padding='same')(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
# ###

    x = tf.keras.layers.Conv1D(64, 5, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(128, 5, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # x = tf.keras.layers.Conv1D(128, 5, 1, padding='same')(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
####

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(vector_output_shape)(x)
    
    # x = tf.linalg.norm(x, axis=-1)
    
    # x = x / tf.reshape(tf.linalg.norm(x, axis=-1), (-1, 1))
    # x = tf.keras.layers.Activation("tanh")(x)
    # tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    
    x = tf.clip_by_norm(x, 1., -1)

    model = tf.keras.Model(_input, x)
    
    # print(model.summary())
    
    # exit()
    
    return model