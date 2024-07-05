import tensorflow as tf
import pandas as pd

def convert_dataframe_to_tensorflow_sequences(
        df:pd.DataFrame, 
        sequence_lenght_in_sample, 
        granularity, 
        shift_between_sequences, 
        shuffle=True) -> tf.data.Dataset:
    
    sequence_lenght = int(sequence_lenght_in_sample*granularity)

    dset = tf.data.Dataset.from_tensor_slices(df.values)
    dset = dset.window(sequence_lenght , shift=shift_between_sequences, stride=granularity).flat_map(lambda x: x.batch(sequence_lenght_in_sample, drop_remainder=True))

    if shuffle:
        dset= dset.shuffle(256)

    return dset


def make_train_valid_dset(
        df:pd.DataFrame, sequence_lenght_in_sample:int, 
        granularity:int, 
        shift_between_sequences:int, 
        train_batch_size:int,
        valid_batch_size:int, 
        shuffle=True,
        valid_set_size:int=5000,
        reduce_train_set:bool=False,
        reduction_coeficient:int=500) -> tf.data.Dataset:
    
    sequence_lenght = int(sequence_lenght_in_sample*granularity)

    dset = tf.data.Dataset.from_tensor_slices(df.values)
    dset = dset.window(sequence_lenght , shift=shift_between_sequences, stride=granularity).flat_map(lambda x: x.batch(sequence_lenght_in_sample, drop_remainder=True))

    train_dset = dset.skip(valid_set_size)
    valid_dset = dset.take(valid_set_size)

    if reduce_train_set == True:
        print("[+] Reducing Train set size...")
        train_dset = train_dset.take(reduction_coeficient* train_batch_size)

    if shuffle:
        train_dset= train_dset.shuffle(20000)

    if train_batch_size > 0:
        train_dset = train_dset.batch(train_batch_size, drop_remainder=True)
        valid_dset = valid_dset.batch(valid_batch_size, drop_remainder=True)

    return train_dset, valid_dset