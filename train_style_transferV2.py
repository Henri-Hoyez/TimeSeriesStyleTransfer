### Train the first version of the time series style transfer.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import argparse

from utils.gpu_memory_grow import gpu_memory_grow
from configs.mts_style_transfer_v2.args import DafaultArguments as args
from utils import dataLoader
from algorithms.mts_style_transferv2 import Trainer

gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)

def parse_arguments():
    default_args = args()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--content_dset", 
        help='The content dataset path', 
        default=default_args.content_dataset_path
    )
    
    parser.add_argument(
        "--style_datasets", 
        help='Styles Datasets', nargs='+', 
        default=default_args.style_datasets_path
    )
    
    parser.add_argument(
        "--epochs",
        help='Number of epochs', type=int,
        default=default_args.simulated_arguments.epochs)

    parser.add_argument(
        "--tensorboard_root", 
        help='The root folder for tensorflow', 
        default=default_args.tensorboard_root_folder
    )
    
    parser.add_argument(
        "--exp_folder", 
        help='Helpfull for grouping experiment', 
        default=default_args.experiment_folder
    )

    parser.add_argument(
        "--exp_name", 
        help='The name of the experiment ;D', 
        default=default_args.exp_name
    )
    
    parser.add_argument(
        '--save_to', 
        help='The folder where the model will be saved.', 
        default=default_args.default_root_save_folder
    )

    arguments = parser.parse_args()

    return arguments



def main():
    shell_arguments = parse_arguments()
    print(shell_arguments)

    standard_arguments = args()
    standard_arguments.simulated_arguments.epochs = shell_arguments.epochs

    sequence_length = standard_arguments.simulated_arguments.sequence_lenght_in_sample
    gran = standard_arguments.simulated_arguments.granularity
    overlap = standard_arguments.simulated_arguments.overlap
    bs = standard_arguments.simulated_arguments.batch_size
    ###
    
    content_dset_train, content_dset_valid = dataLoader.loading_wrapper(
        shell_arguments.content_dset,
        sequence_length, 
        gran, 
        overlap, 
        2*bs) # Two Times BS for the training function.
    
    # Load Styles:
    style_dsets_train, style_dsets_valid = [], []
    style_seeds_train, style_seeds_valid = [], []
    for i, style_path in enumerate(shell_arguments.style_datasets):
        style_labels = tf.zeros((1,)) + i

        style_train, style_valid =  dataLoader.loading_wrapper(
            style_path, 
            sequence_length, 
            gran, 
            overlap,
            0)
        
        _style_seed_train = dataLoader.get_batches(style_train.batch(bs), 50)
        _style_seed_valid = dataLoader.get_batches(style_valid.batch(bs), 50)

        style_seeds_train.append(_style_seed_train)
        style_seeds_valid.append(_style_seed_valid)

        style_train = style_train.map(lambda seq: (seq, style_labels))
        style_valid = style_valid.map(lambda seq: (seq, style_labels))

        style_dsets_train.append(style_train)
        style_dsets_valid.append(style_valid)
        
    style_seeds_train = tf.convert_to_tensor(style_seeds_train)
    style_seeds_valid = tf.convert_to_tensor(style_seeds_valid)

    style_dsets_train = tf.data.Dataset.sample_from_datasets(style_dsets_train).batch(bs, drop_remainder=True)
    style_dsets_valid = tf.data.Dataset.sample_from_datasets(style_dsets_valid).batch(bs, drop_remainder=True)

    trainner = Trainer(shell_arguments, standard_arguments)

    trainner.instanciate_datasets(
        content_dset_train, content_dset_valid,
        style_dsets_train, style_dsets_valid,
    )

    trainner.set_seeds(style_seeds_train, style_seeds_valid)

    trainner.train()


if __name__ == "__main__":
    main()