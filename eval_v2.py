import tensorflow as tf
import argparse
import numpy as np
from models.evaluation import utils

test = tf.keras.layers.Dense(100)
from utils.gpu_memory_grow import gpu_memory_grow

gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_folder", 
        help='The model where weight are saved.'
    )
    
    return parser.parse_args()
    
    
def stylize(
        content_dset:tf.data.Dataset, 
        style_sequence:tf.Tensor, 
        content_encoder:tf.keras.Model, 
        style_encoder:tf.keras.Model, 
        decoder:tf.keras.Model) -> tf.data.Dataset:
    
    content_space = content_dset.map(lambda seq, l: (content_encoder(seq), l), num_parallel_calls=tf.data.AUTOTUNE).cache()
    
    style_vector = style_encoder(style_sequence)

    return content_space.map(lambda c, l: (tf.concat(decoder([c, style_vector], training=False), -1), l), num_parallel_calls=tf.data.AUTOTUNE).cache()

    
    
    
def generate_datasets(training_arguments:dict, ce, se, de):
    bs = 256 #args().simulated_arguments.batch_size
    real_style_dataset = {}
    fake_style_dataset = {}
    style_names = []

    dset_content_train, dset_content_valid = utils.load_dset(
        training_arguments["dset_content"], 
        training_arguments, drop_labels=False, bs=bs)

    dset_content_train = utils.extract_labels(dset_content_train, training_arguments)
    dset_content_valid = utils.extract_labels(dset_content_valid, training_arguments)

    for style_path in training_arguments["style_datasets"]:
        sty_name = utils.get_name(style_path)
        style_names.append(sty_name)
        
        print(f"Making {sty_name}")
        
        dset_style_train, dset_style_valid = utils.load_dset(style_path, training_arguments, drop_labels=False, bs=bs)
        
        dset_lstyle_train = utils.extract_labels(dset_style_train, training_arguments)
        dset_lstyle_valid = utils.extract_labels(dset_style_valid, training_arguments)
            
        real_style_dataset[f"{sty_name}_train"] = dset_lstyle_train
        real_style_dataset[f"{sty_name}_valid"] = dset_lstyle_valid
        
        style_batch_train = next(iter(dset_lstyle_train))[0][0]
        style_batch_train = np.array([style_batch_train]* bs)
        
        style_batch_valid = next(iter(dset_lstyle_valid))[0][0]
        style_batch_valid = np.array([style_batch_valid]* bs)
                
        stylized_train = stylize(dset_content_train, style_batch_train, ce, se, de)
        stylized_valid = stylize(dset_content_valid, style_batch_valid, ce, se, de)
        
        fake_style_dataset[f"{sty_name}_train"] = stylized_train
        fake_style_dataset[f"{sty_name}_valid"] = stylized_valid  
        
    return real_style_dataset, fake_style_dataset

def main():
    shell_args = parse_arguments()
    print(shell_args)
    
    training_arguments = utils.get_model_training_arguments(shell_args.model_folder)
    ce, se, de = utils.load_models(shell_args.model_folder)
    print()
    print(training_arguments['sequence_lenght_in_sample'])
        
    real_dict, fake_dict = generate_datasets(training_arguments, ce, se, de)
    exit()  
    
    
    
    
    
    
    
    



if __name__ == '__main__':
    main()