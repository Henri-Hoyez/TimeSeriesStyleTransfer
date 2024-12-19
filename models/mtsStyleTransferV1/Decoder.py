import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
# import tensorflow as tf

# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Layer
# from tensorflow.python.keras.initializers import RandomNormal
# from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Dense, Conv1DTranspose, Reshape, Flatten
# from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import BatchNormalization, SpectralNormalization # type: ignore
from tensorflow.keras import Input # type: ignore 
from tensorflow.keras.layers import SpectralNormalization # type: ignore
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate, Conv1DTranspose, Reshape# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.regularizers import L2 # type: ignore
from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras.initializers import RandomNormal 

from tensorflow.keras.layers import Add

from models.Layers.AdaIN import AdaIN


def linear_projection(style_input:Layer, actual_seq_len:int):

    adapter = Dense(actual_seq_len)(style_input)
    adapter = Reshape((actual_seq_len, 1))(adapter)

    return adapter

def upsampling_block(content_input:Layer, ec_features, style_input:Layer, filters):
    
    _input = Concatenate(axis=-1)([content_input, ec_features])
    
    print(content_input.shape, ec_features.shape, _input.shape)

    x_first = Conv1DTranspose(filters, 5, 1, padding='same')(_input)
    actual_sequence_len = x_first.shape[1]
    
    adapted_style_input = linear_projection(style_input, actual_sequence_len)
    x = AdaIN()(x_first, adapted_style_input)
    x = LeakyReLU()(x)

    x = Conv1DTranspose(filters, 5, 1, padding='same')(x)
    adapted_style_input = linear_projection(style_input, actual_sequence_len)
    x = AdaIN()(x, adapted_style_input)
    x = LeakyReLU()(x)
    
    # x = Add()([x_first, x])

    x = Conv1DTranspose(filters, 5, 2, padding='same')(x)
    adapted_style_input = linear_projection(style_input, actual_sequence_len*2)
    x = AdaIN()(x, adapted_style_input)
    x = LeakyReLU()(x)   
    
    return x


def generator_part(content_inputs, style_input):

    [d1, d2, d3, x] = content_inputs

    x = upsampling_block(x, d3, style_input, 16) # 16

    x = upsampling_block(x, d2, style_input, 32) # 32
    
    x = upsampling_block(x, d1, style_input, 64)  # 64
    
    # output
    x = Conv1DTranspose(1, 5, 1, padding='same')(x)
    
    return x

def make_generator(n_sample_wiener:int, feat_wiener:int, style_vector_size:int, n_generators:int, ec_outputs: list):
    init = RandomNormal()

    content_inputs = [Input(eci.shape[1:]) for eci in ec_outputs]
    style_input = Input((style_vector_size,), name="Style_Input") 
    gens_outputs = []


    for _ in range(n_generators):
        gens_outputs.append(generator_part(content_inputs, style_input))
        # break

    model = Model([content_inputs, style_input], gens_outputs)
    
    model.summary()

    return model

