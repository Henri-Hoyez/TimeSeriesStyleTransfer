import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from configs.RealData import Proposed

config = Proposed()

def standardize_dataframes(dfs:tuple, mean_scaling:bool):
    if mean_scaling:
        scaler = StandardScaler()
    else: 
        scaler = MinMaxScaler()
        
    df = pd.concat(dfs)
    scaler.fit(df.values)

    scaled_dfs_values = [scaler.transform(dfi.values) for dfi in dfs]
    scaled_dfs = (pd.DataFrame(scaled_dfi_values, dfi.index, dfi.columns) for scaled_dfi_values, dfi in zip(scaled_dfs_values, dfs))
    return scaled_dfs


def is_nan(sequence):
    return tf.reduce_sum(
        tf.cast(
            tf.math.is_nan(sequence),
            tf.int32,
        )
    ) == tf.constant(0)
    
    
def convert_dataframe_to_tensorflow_sequences(df:pd.DataFrame, sequence_lenght_in_sample, granularity, shift_between_sequences, batch_size, shuffle=True):
    sequence_lenght = int(sequence_lenght_in_sample*granularity)

    dset = tf.data.Dataset.from_tensor_slices(df.values)
    dset = dset.window(sequence_lenght , shift=shift_between_sequences, stride=granularity).flat_map(lambda x: x.batch(sequence_lenght_in_sample, drop_remainder=True))

    dset = dset.filter(is_nan)

    if shuffle:
        dset= dset.shuffle(256)

    dset = dset.batch(batch_size, drop_remainder=True)

    dset = dset.cache().prefetch(10)    

    return dset