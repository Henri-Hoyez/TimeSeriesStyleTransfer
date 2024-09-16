import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

from models.Layers.AdaIN import AdaIN

def generator_part(content_input, n_sample_wiener:int, feat_wiener:int, style_input):
    init = tf.keras.initializers.RandomNormal()

    # Make a small projection...
    _content_input = tf.keras.layers.Flatten()(content_input)
    _content_input = tf.keras.layers.Dense(n_sample_wiener* feat_wiener, kernel_initializer=init)(_content_input)
    _content_input = tf.keras.layers.Reshape((n_sample_wiener, feat_wiener))(_content_input)

    # Make the style input 
    _style_input = tf.keras.layers.Dense(16)(style_input)
    _style_input = tf.keras.layers.Reshape((16, 1))(_style_input)

    _stage2_style_input = tf.keras.layers.Dense(32)(style_input)
    _stage2_style_input = tf.keras.layers.Reshape((32, 1))(_stage2_style_input)


    x = tf.keras.layers.Conv1DTranspose(16, 5, 1, padding='same', kernel_initializer=init)(_content_input)
    x = AdaIN()(x, _style_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1DTranspose(16, 5, 1, padding='same', kernel_initializer=init)(x)
    x = AdaIN()(x, _style_input)
    x = tf.keras.layers.LeakyReLU()(x)  

    x = tf.keras.layers.Conv1DTranspose(16, 5, 2, padding='same', kernel_initializer=init)(x)
    # x = AdaIN()(x, _stage2_style_input) 
    x = tf.keras.layers.LeakyReLU()(x)


    x = tf.keras.layers.Conv1DTranspose(32, 5, 1, padding='same', kernel_initializer=init)(x)
    x = AdaIN()(x, _stage2_style_input) 
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1DTranspose(32, 5, 1, padding='same', kernel_initializer=init)(x)
    x = AdaIN()(x, _stage2_style_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1DTranspose(32, 5, 2, padding='same', kernel_initializer=init)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # output
    x = tf.keras.layers.Conv1DTranspose(1, 5, 1, padding='same', kernel_initializer=init)(x)
    
    # x = tf.keras.layers.LeakyReLU()(x)

    return x

def make_generator(n_sample_wiener:int, feat_wiener:int, style_vector_size:int, n_generators:int)  -> tf.keras.Model :

    input = tf.keras.Input((n_sample_wiener, feat_wiener), name=f"Content_Input")
    style_input = tf.keras.Input((style_vector_size,), name="Style_Input") 
    gens_outputs = []

    for _ in range(n_generators):
        gens_outputs.append(generator_part(input, n_sample_wiener, feat_wiener, style_input))

    model = tf.keras.Model([input, style_input], gens_outputs)

    return model