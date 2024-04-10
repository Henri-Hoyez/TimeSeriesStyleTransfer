import tensorflow as tf
import pandas as pd

def convert_dataframe_to_tensorflow_sequences(df:pd.DataFrame, sequence_lenght_in_sample, granularity, shift_between_sequences, batch_size, shuffle=True):
    sequence_lenght = int(sequence_lenght_in_sample*granularity)

    dset = tf.data.Dataset.from_tensor_slices(df.values)
    dset = dset.window(sequence_lenght , shift=shift_between_sequences, stride=granularity).flat_map(lambda x: x.batch(sequence_lenght_in_sample, drop_remainder=True))

    if shuffle:
        dset= dset.shuffle(256)

    dset = dset.batch(batch_size, drop_remainder=True)

    dset = dset.cache().prefetch(10)

    return dset