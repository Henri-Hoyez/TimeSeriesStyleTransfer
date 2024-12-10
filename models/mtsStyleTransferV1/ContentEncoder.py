import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Conv1D, LeakyReLU
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.regularizers import L2

from tensorflow.keras.layers import BatchNormalization, SpectralNormalization # type: ignore
from tensorflow.keras import Input # type: ignore 
from tensorflow.keras.layers import SpectralNormalization # type: ignore
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate, Layer# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.regularizers import L2, L1 # type: ignore


def downsampling_block(input:Layer, filters:int) -> Layer:
    x = Conv1D(filters, 2, 2, padding='same')(input)# , kernel_regularizer=regularizers.l2(0.001)   
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv1D(filters, 3, 1, padding='same')(x)# , kernel_regularizer=regularizers.l2(0.001)   
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    return x


def make_content_encoder(seq_length:int, n_feat:int, feat_wiener:int):

    _input = Input((seq_length, n_feat))

    x = downsampling_block(_input, 16)
    
    x = downsampling_block(x, 32)
    
    x = downsampling_block(x, 64)
    
    x = Conv1D(2, 1, 1, padding='same', activation="linear")(x)

    model = Model(_input, x)
    
    # model.summary()
    # exit()
    
    return model




# def make_content_encoder(seq_length:int, n_feat:int, feat_wiener:int):

#     _input = Input((seq_length, n_feat))

#     x = Conv1D(32, 5, 2, padding='same')(_input)# , kernel_regularizer=regularizers.l2(0.001)   
#     x = LeakyReLU()(x)
    
#     x =  Conv1D(64, 5, 2, padding='same')(x) # , kernel_regularizer=regularizers.l2(0.001)   
#     x = LeakyReLU()(x)
    
#     x = Conv1D(128, 5, 2, padding='same')(x)# , kernel_regularizer=regularizers.l2(0.001)   
#     x = LeakyReLU()(x)

#     x = Conv1D(feat_wiener, 5, 1, padding='same', activation="linear")(x)

#     model = Model(_input, x)
    
#     return model