import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import numpy as np


def load_dataframe(df_path:str, drop_labels=True):
    _df= pd.read_hdf(df_path).astype(np.float32)

    if drop_labels == True:
        _df = _df.drop(columns=['labels'])

    return _df


def train_valid_split(df:pd.DataFrame):
    train_size = 0.8
    train_last_sample = int(df.shape[0]* train_size)
    _df = df.copy()

    train_df = _df.iloc[:train_last_sample]
    valid_df = _df.iloc[train_last_sample:]

    return train_df, valid_df



def pd2tf(df:pd.DataFrame, sequence_lenght, granularity, overlap, batch_size, shuffle:bool):
    total_seq_len = int(sequence_lenght* granularity)
    shift_between_sequences = int(total_seq_len* overlap)

    dset = tf.data.Dataset.from_tensor_slices(df.values)
    dset = dset.window(sequence_lenght , shift=shift_between_sequences, stride=granularity).flat_map(lambda x: x.batch(sequence_lenght, drop_remainder=True))

    if shuffle == True:
        dset = dset.shuffle(2000)

    if batch_size > 0:
        dset = dset.batch(batch_size, drop_remainder=True)

    return dset


def remove_format(path:str):
    return ".".join(path.split('.')[:-1])

def loading_wrapper(df_path:str, sequence_lenght:int, granularity:int, overlap:int, batch_size:int, shuffle:bool=True, drop_labels:bool=True):
    
    path_placeholder = remove_format(df_path)
    
    train_path = f"{path_placeholder}_train.h5"
    valid_path = f"{path_placeholder}_valid.h5"
    
    _df_train = load_dataframe(train_path, drop_labels)
    _df_valid = load_dataframe(valid_path, drop_labels)

    _dset_train = pd2tf(_df_train, sequence_lenght, granularity, overlap, batch_size, shuffle)
    _dset_valid = pd2tf(_df_valid, sequence_lenght, granularity, overlap, batch_size, False)

    return _dset_train, _dset_valid


def get_batches(dset, n_batches):
    _arr = np.array([c for c in dset.take(n_batches)])
    return _arr.reshape((-1, _arr.shape[-2], _arr.shape[-1]))


def get_seed_visualization_content_sequences(content_path:str, sequence_len:int):
    path_placeholder = remove_format(content_path)
    
    valid_path = f"{path_placeholder}_valid.h5"
    
    _df_valid = load_dataframe(valid_path, False)
    labels = _df_valid['labels'].unique()
    
    content_sequences = []
    
    for l in labels:
        df_part = _df_valid[_df_valid["labels"] == l]
        
        indexes = df_part.index
        
        start_index = indexes[0]
        end_index= start_index+ sequence_len

        content_sequence = _df_valid.loc[start_index: end_index-1].values[:, :-1]
        content_sequences.append(content_sequence)
        
        # plt.figure(figsize=(18, 10))
        # plt.plot(content_sequence)
        # plt.savefig(f"{l}.png")
        
    return content_sequences