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
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dense, Dropout, Flatten, Concatenate# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.regularizers import L2 # type: ignore




def make_content_encoder(seq_length:int, n_feat:int, feat_wiener:int):

    _input = Input((seq_length, n_feat))

    x = Conv1D(16, 5, 2, padding='same')(_input)# , kernel_regularizer=regularizers.l2(0.001)   
    x = LeakyReLU()(x)
    
    x =  Conv1D(32, 5, 2, padding='same')(x) # , kernel_regularizer=regularizers.l2(0.001)   
    x = LeakyReLU()(x)
    
    x = Conv1D(64, 5, 2, padding='same')(x)# , kernel_regularizer=regularizers.l2(0.001)   
    x = LeakyReLU()(x)

    x = Conv1D(feat_wiener, 5, 1, padding='same', activation="linear")(x)

    model = Model(_input, x)
    

    return model