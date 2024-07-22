### Train the first version of the time series style transfer.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import argparse

from utils.metric import signature_on_batch, signature_metric
from utils.DataManagement import  pd_to_tf_dset
from models.mts_style_transfer import make_generator, make_content_encoder, make_style_encoder, make_global_discriminator, create_local_discriminator
from utils.simple_metric import simple_metric_on_noise
from utils.gpu_memory_grow import gpu_memory_grow
from configs.mts_style_transfer_v1.args import AmplitudeShiftArgs as args
from utils import dataLoader
from algorithms.mts_style_transferv1 import Trainer

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
        "--style1_dataset", 
        help='Style 1 dataset Path', 
        default=default_args.style1_dataset_path
    )
    
    parser.add_argument(
        "--style2_dataset", 
        help='Style 2 dataset Path', 
        default=default_args.style2_dataset_path
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
    # print(arguments)

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
    
    style1_dset_train, style1_dset_valid =  dataLoader.loading_wrapper(
        shell_arguments.style1_dataset, 
        sequence_length, 
        gran, 
        overlap,
        bs)
    
    style2_dset_train, style2_dset_valid =  dataLoader.loading_wrapper(
        shell_arguments.style2_dataset, 
        sequence_length, 
        gran, 
        overlap,
        bs)
    
    trainner = Trainer(shell_arguments, standard_arguments)

    trainner.instanciate_datasets(
        content_dset_train, content_dset_valid,
        style1_dset_train, style1_dset_valid,
        style2_dset_train, style2_dset_valid
    )

    trainner.train()


if __name__ == "__main__":
    main()