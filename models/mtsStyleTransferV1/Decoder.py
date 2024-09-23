import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

from models.Layers.AdaIN import AdaIN



def linear_projection(style_input:tf.keras.layers.Layer, actual_seq_len:int):

    adapter = tf.keras.layers.Dense(actual_seq_len)(style_input)
    adapter = tf.keras.layers.Reshape((actual_seq_len, 1))(adapter)

    return adapter

def upsampling_block(content_input:tf.keras.layers.Layer, style_input:tf.keras.layers.Layer, filters):

    actual_sequence_len = content_input.shape[1]

    x = tf.keras.layers.Conv1D(filters, 5, 1, padding='same')(content_input)
    adapted_style_input = linear_projection(style_input, actual_sequence_len)
    x = AdaIN()(x, adapted_style_input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters, 5, 1, padding='same')(x)
    adapted_style_input = linear_projection(style_input, actual_sequence_len)
    x = AdaIN()(x, adapted_style_input)
    x = tf.keras.layers.LeakyReLU()(x)  

    # x = tf.keras.layers.Conv1D(filters, 5, 1, padding='same')(x)
    # adapted_style_input = linear_projection(style_input, actual_sequence_len)
    # x = AdaIN()(x, adapted_style_input)
    # x = tf.keras.layers.LeakyReLU()(x)  

    # x = tf.keras.layers.Conv1D(filters, 5, 1, padding='same')(x)
    # adapted_style_input = linear_projection(style_input, actual_sequence_len)
    # x = AdaIN()(x, adapted_style_input)
    # x = tf.keras.layers.LeakyReLU()(x)  

    x = tf.keras.layers.Conv1DTranspose(filters, 5, 2, padding='same')(x)
    # x = tf.keras.layers.LeakyReLU()(x)

    return x


def generator_part(content_input, style_input):

    x = upsampling_block(content_input, style_input, 64) # 16

    x = upsampling_block(x, style_input, 32) # 32
    
    x = upsampling_block(x, style_input, 32)  # 64
    
    x = upsampling_block(x, style_input, 32)  # 128
    
    # x = upsampling_block(x, style_input, 32)  # 256
    # output
    x = tf.keras.layers.Conv1DTranspose(1, 5, 1, padding='same')(x)

    return x

def make_generator(n_sample_wiener:int, feat_wiener:int, style_vector_size:int, n_generators:int):
    init = tf.keras.initializers.RandomNormal()

    input = tf.keras.Input((n_sample_wiener, feat_wiener), name=f"Content_Input")
    style_input = tf.keras.Input((style_vector_size,), name="Style_Input") 
    gens_outputs = []


    # Same for the content, project into the same space.
        # Make a small projection for the content.
    # _content_input = tf.keras.layers.Flatten()(input)
    # _content_input = tf.keras.layers.Dense(n_sample_wiener* feat_wiener, kernel_initializer=init)(_content_input)
    # _content_input = tf.keras.layers.Reshape((n_sample_wiener, feat_wiener))(_content_input)

    for _ in range(n_generators):
        gens_outputs.append(generator_part(input, style_input))
        # break

    model = tf.keras.Model([input, style_input], gens_outputs)
    
    

    return model

