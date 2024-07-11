import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import argparse

from configs.mts_style_transfer_v1.args import AmplitudeShiftArgs as args
from models.Layers.AdaIN import AdaIN
from utils import eval_methods, dataLoader
from configs.mts_style_transfer_v1.args import AmplitudeShiftArgs as args





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
        "--exp_folder",
        help="the folder where the experiments are located.",
        default=default_args.default_root_save_folder
    )

    parser.add_argument(
        "--exp_name", 
        help='The name of the experiment ;D', 
        default=default_args.exp_name
    )

    arguments = parser.parse_args()

    return arguments


def load_models(shell_arguments:dict):
    folder = f"{shell_arguments.exp_folder}/{shell_arguments.exp_name}"

    content_encoder = tf.keras.models.load_model(f"{folder}/content_encoder.h5", custom_objects={'AdaIN':AdaIN})
    style_encoder = tf.keras.models.load_model(f"{folder}/style_encoder.h5", custom_objects={'AdaIN':AdaIN})
    decoder = tf.keras.models.load_model(f"{folder}/decoder.h5", custom_objects={'AdaIN':AdaIN})

    return content_encoder, style_encoder, decoder


def load_valid_datasets(shell_arguments:dict):
    default_parameters = args()

    default_parameters.simulated_arguments.epochs = shell_arguments.epochs

    sequence_length = default_parameters.simulated_arguments.sequence_lenght_in_sample
    gran = default_parameters.simulated_arguments.granularity
    overlap = default_parameters.simulated_arguments.overlap
    bs = default_parameters.simulated_arguments.batch_size

    _, content_dset_valid = dataLoader.loading_wrapper(
        shell_arguments.content_dset,
        sequence_length, 
        gran, 
        overlap, 
        2*bs)
    
    _, style1_dset_valid =  dataLoader.loading_wrapper(
        shell_arguments.style1_dataset, 
        sequence_length, 
        gran, 
        overlap,
        bs)
    
    _, style2_dset_valid =  dataLoader.loading_wrapper(
        shell_arguments.style2_dataset, 
        sequence_length, 
        gran, 
        overlap,
        bs)
    
    return content_dset_valid, style1_dset_valid, style2_dset_valid


def main():
    shell_arguments = parse_arguments()

    dset_content_valid, dset_style1_valid, dset_style2_valid = load_valid_datasets(shell_arguments)
    content_encoder, style_encoder, decoder = load_models(shell_arguments)








if __name__ == "__main__":
    main()