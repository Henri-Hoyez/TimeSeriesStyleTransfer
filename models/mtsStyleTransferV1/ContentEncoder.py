import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

# def make_content_encoder(seq_length:int, n_feat:int, feat_wiener:int) -> tf.keras.Model:

#     _input = tf.keras.Input((seq_length, n_feat))

#     x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(_input)
#     # x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
        
    
#     x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(x)
#     # x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
    
#     x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(x)
#     # x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
    
#     # x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(x)
#     # # x = tf.keras.layers.BatchNormalization()(x)
#     # x = tf.keras.layers.LeakyReLU()(x)


    
#     x = tf.keras.layers.Conv1D(feat_wiener, 5, 1, padding='same', activation="linear")(x)
    
    
#     # x = tf.math.l2_normalize(x, -1)
#     # x = tf.keras.layers.LayerNormalization()(x)
    

#     model = tf.keras.Model(_input, x)
    
#     # print(model.summary())
#     # exit()


#     return model


def make_content_encoder(seq_length:int, n_feat:int, feat_wiener:int):

    _input = tf.keras.Input((seq_length, n_feat))

    x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(_input)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
        
    
    x = tf.keras.layers.Conv1D(32, 5, 2, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv1D(64, 5, 2, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)


    # x = tf.keras.layers.Conv1D(16, 5, 2, padding='same')(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)

    
    x = tf.keras.layers.Conv1D(feat_wiener, 5, 1, padding='same', activation="linear")(x)
    
    x = tf.clip_by_norm(x, 1., -1)
    
    # x = tf.math.l2_normalize(x, -1)
    # x = tf.keras.layers.LayerNormalization()(x)
    

    model = tf.keras.Model(_input, x)
    
    # print(model.summary())
    
    # exit()
    return model