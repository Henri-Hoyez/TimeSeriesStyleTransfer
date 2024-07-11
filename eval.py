import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
import argparse

from configs.mts_style_transfer_v1.args import AmplitudeShiftArgs as args
from models.Layers.AdaIN import AdaIN
from utils import dataLoader, simple_metric


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

def generate(
        content_batch, 
        style_batch, 
        content_encoder, 
        style_encoder, 
        decoder):
    content = content_encoder(content_batch, training=False)
    style = style_encoder(style_batch, training=False)
    generated = decoder([content, style], training=False)
    return generated


def load_valid_batches(shell_arguments:dict):
    default_parameters = args()

    sequence_length = default_parameters.simulated_arguments.sequence_lenght_in_sample
    gran = default_parameters.simulated_arguments.granularity
    overlap = default_parameters.simulated_arguments.overlap
    bs = default_parameters.simulated_arguments.batch_size

    _, content_dset_valid = dataLoader.loading_wrapper(
        shell_arguments.content_dset,
        sequence_length, 
        gran, 
        overlap, 
        bs)
    
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
    
    content_batch = dataLoader.get_batches(content_dset_valid, 100)
    style1_batch = dataLoader.get_batches(style1_dset_valid, 100) 
    style2_batch = dataLoader.get_batches(style2_dset_valid, 100) 
    
    return content_batch, style1_batch, style2_batch


def evaluate():
    print('hello !!!')
    result_dictionary = dict()

    shell_arguments = parse_arguments()

    content_big_batch, style1_big_batch, style2_big_batch = load_valid_batches(shell_arguments)
    
    content_encoder, style_encoder, decoder = load_models(shell_arguments)

    
    generated_style1 = generate(content_big_batch, style1_big_batch,
                                content_encoder, style_encoder, decoder)
    
    generated_style2 = generate(content_big_batch, style2_big_batch,
                                content_encoder, style_encoder, decoder)

    _, content_extracted_noise = simple_metric.simple_metric_on_noise(content_big_batch)
    _, style1_extracted_noise = simple_metric.simple_metric_on_noise(style1_big_batch)
    _, style2_extracted_noise = simple_metric.simple_metric_on_noise(style2_big_batch)

    _, gen_s1_extracted_noise = simple_metric.simple_metric_on_noise(generated_style1)
    _, gen_s2_extracted_noise = simple_metric.simple_metric_on_noise(generated_style2)
    
    result_dictionary["content_extracted_noise"]= content_extracted_noise
    result_dictionary["style1_extracted_noise"] = style1_extracted_noise
    result_dictionary["style2_extracted_noise"] = style2_extracted_noise
    result_dictionary["gen_s1_extracted_noise"] = gen_s1_extracted_noise
    result_dictionary["gen_s2_extracted_noise"] = gen_s2_extracted_noise

    content_extracted_ampl = simple_metric.extract_amplitude_from_signals(content_big_batch)
    style1_extracted_ampl = simple_metric.extract_amplitude_from_signals(style1_big_batch)
    style2_extracted_ampl = simple_metric.extract_amplitude_from_signals(style2_big_batch)
    
    gen_s1_extracted_ampl = simple_metric.extract_amplitude_from_signals(generated_style1)
    gen_s2_extracted_ampl = simple_metric.extract_amplitude_from_signals(generated_style2)

    result_dictionary['content_extracted_ampl'] = content_extracted_ampl
    result_dictionary['style1_extracted_ampl'] = style1_extracted_ampl
    result_dictionary['style2_extracted_ampl'] = style2_extracted_ampl
    result_dictionary['gen_s1_extracted_ampl'] = gen_s1_extracted_ampl
    result_dictionary['gen_s2_extracted_ampl'] = gen_s2_extracted_ampl

    df = pd.DataFrame().from_dict(result_dictionary)

    df.to_excel(f"{shell_arguments.exp_folder}/{shell_arguments.exp_name}.xlsx")
    

if __name__ == "__main__":
    evaluate()