import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.keras.layers.Dense(100)
from utils.gpu_memory_grow import gpu_memory_grow

import matplotlib.pyplot as plt
import itertools
from models.evaluation import utils
from utils import eval_methods, dataLoader, simple_metric, dataLoader

# from configs.mts_style_transfer_v2.args import DafaultArguments as args
from configs.mts_style_transfer_v2.args_sim import DafaultArguments as args
# from configs.mts_style_transfer_v2.args_real import DafaultArguments as args

from tensorflow.python.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.python.keras.models import Model

from models.NaiveClassifier import make_naive_discriminator


import umap
from sklearn.manifold import TSNE
import argparse
from utils import visualization_helpersv2
from utils import visualization_helpersv2
from sklearn.decomposition import PCA



gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_folder", 
        help='The folder where the trained model is saved.', 
    )
    
    return parser.parse_args()

def get_name(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])

def stylize(
        content_dset:tf.data.Dataset, 
        style_sequence:tf.Tensor, 
        content_encoder:tf.keras.Model, 
        style_encoder:tf.keras.Model, 
        decoder:tf.keras.Model) -> tf.data.Dataset:
    
    content_space = content_dset.map(lambda seq, _: (content_encoder(seq)), num_parallel_calls=tf.data.AUTOTUNE).cache()
    labels = content_dset.map(lambda _,l : l, num_parallel_calls=tf.data.AUTOTUNE).cache()
    
    style_vector = style_encoder(style_sequence)

    translated = content_space.map(lambda c: tf.concat(decoder([c, style_vector], training=False), -1), num_parallel_calls=tf.data.AUTOTUNE).cache()
    dset_final = tf.data.Dataset.zip((translated, labels))

    return dset_final

def generate_real_fake_datasets(training_params, ce, se, de):
    real_style_dataset = {}
    fake_style_dataset = {}
    style_names = []
    bs = 256 #args().simulated_arguments.batch_size

    dset_content_train, dset_content_valid = utils.load_dset(training_params["dset_content"], training_params, drop_labels=False, bs=bs)

    dset_content_train = utils.extract_labels(dset_content_train, training_params)
    dset_content_valid = utils.extract_labels(dset_content_valid, training_params)

    for style_path in training_params["style_datasets"]:
        sty_name = get_name(style_path)
        style_names.append(sty_name)
        
        print(f"Making {sty_name}")
        
        dset_style_train, dset_style_valid = utils.load_dset(style_path, training_params, drop_labels=False, bs=bs)
        
        dset_lstyle_train = utils.extract_labels(dset_style_train, training_params)
        dset_lstyle_valid = utils.extract_labels(dset_style_valid, training_params)
            
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

def tstr(
    dset_train_real,
    dset_valid_real,
    dset_train_fake, 
    dset_valid_fake, 
    save_to:str):

    print('[+] Train Real, Test Real.')
    trtr_perfs, trtr_hist = eval_methods.train_naive_discriminator(dset_train_real, dset_valid_real, args(), epochs=50, n_classes=5)

    print("[+] Train Synthetic, Test Synthetic")
    _, tsts_hist = eval_methods.train_naive_discriminator(dset_train_fake, dset_valid_fake, args(), epochs=50, n_classes=5)
    
    print("[+] Train Synthetic, Test Real")
    tstr_perfs, tstr_hist = eval_methods.train_naive_discriminator(dset_train_fake, dset_valid_real, args(), epochs=50, n_classes=5)
    
    print("[+] Train Real, Test Synthetic")
    _, trts_hist = eval_methods.train_naive_discriminator(dset_train_real, dset_valid_fake, args(), epochs=50, n_classes=5)
    
    fig = plt.figure(figsize=(18, 10))
    
    ax = plt.subplot(421)
    ax.set_title("Train Real Test Real loss")
    
    plt.plot(trtr_hist.history["loss"], ".-", label='Train')
    plt.plot(trtr_hist.history["val_loss"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    
    ax = plt.subplot(422)
    ax.set_title("Train Real Test Real accuracy")
    
    plt.plot(trtr_hist.history["sparse_categorical_accuracy"], ".-", label='Train')
    plt.plot(trtr_hist.history["val_sparse_categorical_accuracy"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    
    #######
    ax = plt.subplot(423)
    ax.set_title("Train Real, Test Synthetic loss")
    
    plt.plot(trts_hist.history["loss"], ".-", label='Train Real, Test Synthetic (Train)')
    plt.plot(trts_hist.history["val_loss"], ".-", label='Train Real, Test Synthetic (Valid)')
    
    ax.grid()
    ax.legend()

    ax = plt.subplot(424)
    ax.set_title("Train Real, Test Synthetic accuracy")
    
    plt.plot(trts_hist.history["sparse_categorical_accuracy"], ".-", label='Train Real, Test Synthetic (Train)')
    plt.plot(trts_hist.history["val_sparse_categorical_accuracy"], ".-", label='Train Real, Test Synthetic (Valid)')
    
    ax.grid()
    ax.legend()
    #######
    
    ax = plt.subplot(425)
    ax.set_title("Train Synthetic, Test Synthetic loss")
    
    plt.plot(tsts_hist.history["loss"], ".-", label='Train')
    plt.plot(tsts_hist.history["val_loss"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    
    ax = plt.subplot(426)
    ax.set_title("Train Synthetic, Test Synthetic accuracy")
    
    plt.plot(tsts_hist.history["sparse_categorical_accuracy"], ".-", label='Train')
    plt.plot(tsts_hist.history["val_sparse_categorical_accuracy"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    #######
    
    ax = plt.subplot(427)
    ax.set_title("Train Synthetic, Test Real loss")
    
    plt.plot(tstr_hist.history["loss"], ".-", label='Train')
    plt.plot(tstr_hist.history["val_loss"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    
    ax = plt.subplot(428)
    ax.set_title("Train Synthetic, Test Real accuracy")
    
    plt.plot(tstr_hist.history["sparse_categorical_accuracy"], ".-", label='Train')
    plt.plot(tstr_hist.history["val_sparse_categorical_accuracy"], ".-", label='Valid')
    ax.grid()
    ax.legend()
    #######
    
    plt.savefig(save_to)
    
    plt.close(fig)
    
    return trtr_perfs, tstr_perfs


def tstr_on_styles(real_dataset, fake_dataset, style_names, model_folder):
    tstr_stats = {}

    for _, style_ in enumerate(style_names):
        print(f'[+] Training on dataset {style_}.')
        
        perf_on_real, perf_on_fake = tstr(
            real_dataset[f"{style_}_train"],
            real_dataset[f"{style_}_valid"],
            fake_dataset[f"{style_}_train"],
            fake_dataset[f"{style_}_valid"], 
            f'{model_folder}/tstr_{style_}.png'
            )
        
        tstr_stats[f"{style_}_real"] = [perf_on_real]
        tstr_stats[f"{style_}_gen"] = [perf_on_fake]
        
    tstr_stats = pd.DataFrame.from_dict(tstr_stats)

    tstr_stats.to_hdf(f"{model_folder}/tstr.h5", key="data")
    
    return tstr_stats

def plot_tstr_results(tstr_stats:pd.DataFrame, model_folder:str):
    def remove_prefix(cols:list):
        return [c.split("_")[0] for c in cols]
    
    tstr_real = tstr_stats.filter(like='real', axis=1)
    tstr_fake = tstr_stats.filter(like='gen', axis=1)

    tstr_real.columns = remove_prefix(tstr_real.columns)
    tstr_fake.columns = remove_prefix(tstr_fake.columns)
    
    plt.figure(figsize=(18, 10))
    ax = plt.subplot(111)
    ax.set_title("Gap Between Accuracies.")

    plt.plot(tstr_real.values.reshape((-1,)), ".-", label='Acc on Real, trained on Real')
    plt.plot(tstr_fake.values.reshape((-1,)), ".-", label='Acc on Real, trained on Fake')

    ax.set_xticklabels(tstr_real.columns.values)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Amplitudes")

    ax.grid(True)
    ax.legend()
    
    plt.savefig(f"{model_folder}/tstr.png")
    
    
def get_batches(dset, n_batches):
    _arr = np.array([c[0] for c in dset.take(n_batches)])
    return _arr.reshape((-1, _arr.shape[-2], _arr.shape[-1]))
    
    
def compute_metrics(dset_real, dset_fake, style_names, model_folder: str):
    def time_shift_evaluation(big_batch):
        return [simple_metric.estimate_time_shift(big_batch, 0, i) for i in range(big_batch.shape[-1])]
    
    real_noise_metric, gen_noise_metric = [], []
    real_ampl_metric, gen_ampl_metric = [], []
    real_ts_metric, gen_ts_metric = [], []

    for style_name in style_names:
        print(f"[+] Compute metric for {style_name}")
        real_batch = get_batches(dset_real[f"{style_name}_valid"], 10)
        fake_batch = get_batches(dset_fake[f"{style_name}_valid"], 10)
        
        real_noise_metric.append(simple_metric.simple_metric_on_noise(real_batch)[-1])
        gen_noise_metric.append(simple_metric.simple_metric_on_noise(fake_batch)[-1])
        
        real_ampl_metric.append(simple_metric.extract_amplitude_from_signals(real_batch))
        gen_ampl_metric.append(simple_metric.extract_amplitude_from_signals(fake_batch))
        
        real_ts_metric.append(time_shift_evaluation(real_batch))
        gen_ts_metric.append(time_shift_evaluation(fake_batch))
        
    real_mean_noises = np.mean(real_noise_metric, axis=-1).reshape((-1, 1))
    fake_mean_noises = np.mean(gen_noise_metric, axis=-1).reshape((-1, 1))
    mean_noises = np.concatenate((real_mean_noises, fake_mean_noises), axis=-1)
    
    real_mean_ampl = np.mean(real_ampl_metric, axis=-1).reshape((-1, 1))
    fake_mean_ampl = np.mean(gen_ampl_metric, axis=-1).reshape((-1, 1))
    mean_ampl= np.concatenate((real_mean_ampl, fake_mean_ampl), axis=-1)
    
    real_mean_time_shift = np.mean(real_ts_metric, axis=-1).reshape((-1, 1))
    fake_mean_time_shift = np.mean(gen_ts_metric, axis=-1).reshape((-1, 1))
    mean_time_shift= np.concatenate((real_mean_time_shift, fake_mean_time_shift), axis=-1)
    
    df_noises = pd.DataFrame(data=mean_noises, index=style_names, columns=['Real', 'Fake'])
    df_ampl = pd.DataFrame(data=mean_ampl, index=style_names, columns=['Real', 'Fake'])
    df_time_shift = pd.DataFrame(data=mean_time_shift, index=style_names, columns=['Real', 'Fake'])
    
    # df_noises.to_hdf(f'{model_folder}/noise_metric.h5', key='data')
    # df_ampl.to_hdf(f'{model_folder}/ampl_metric.h5', key='data')
    # df_time_shift.to_hdf(f'{model_folder}/time_shift_metric.h5', key='data')
    
    df_noises.to_excel(f'{model_folder}/noise_metric.xlsx')
    df_ampl.to_excel(f'{model_folder}/ampl_metric.xlsx')
    df_time_shift.to_excel(f'{model_folder}/time_shift_metric.xlsx')
    
    return df_noises, df_ampl, df_time_shift


def plot_metric(df_metric:pd.DataFrame, title, y_min, y_max, save_to):
    plt.figure(figsize=(18, 10))
    ax = plt.subplot(111)
    
    df_metric["Real"].plot(ax=ax, style='.-')
    df_metric["Fake"].plot(ax=ax, style='.-')
    
    ax.grid(True)
    ax.set_title(title)
    ax.legend()
    
    ax.set_ylim(y_min, y_max)
    
    plt.savefig(save_to)
    
def multi_umap_plot(real_styles, gen_styles):
    (_, _, seq_len, n_sigs) = real_styles.shape
    
    concatenated = tf.concat((real_styles, gen_styles), 0)

    concatenated = tf.reshape(concatenated, (-1, seq_len, n_sigs))
    concatenated = tf.transpose(concatenated, (0, 2, 1))
    
    concatenated = tf.reshape(concatenated, (concatenated.shape[0], -1))

    # # # Normalize all sequences for the reducer.
    _mean, _std = tf.math.reduce_mean(concatenated), tf.math.reduce_std(concatenated)
    concatenated = (concatenated - _mean)/_std

    reducer = umap.UMAP(n_neighbors=300, min_dist=1., random_state=42, metric="euclidean") 
    reduced = reducer.fit_transform(concatenated)
    return reduced


def multi_tsne_plot(real_styles, gen_styles):
    (_, _, seq_len, n_sigs) = real_styles.shape
    
    concatenated = tf.concat((real_styles, gen_styles), 0)

    concatenated = tf.reshape(concatenated, (-1, seq_len, n_sigs))
    concatenated = tf.transpose(concatenated, (0, 2, 1))
    
    concatenated = tf.reshape(concatenated, (concatenated.shape[0], -1))

    # # # Normalize all sequences for the reducer.
    _mean, _std = tf.math.reduce_mean(concatenated), tf.math.reduce_std(concatenated)
    concatenated = (concatenated - _mean)/_std

    reducer = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=150, random_state=42)
    reduced = reducer.fit_transform(concatenated)
    return reduced


def generate_per_style_batch(dset_real, dset_fake, style_names):
    real_batches = []
    fake_batches = []

    for _, style_ in enumerate(style_names):
        real_style_batch = get_batches(dset_real[f"{style_}_valid"], 5)
        fake_style_batch = get_batches(dset_fake[f"{style_}_valid"], 5)
        
        real_batches.append(real_style_batch)
        fake_batches.append(fake_style_batch)
        
    return np.array(real_batches), np.array(fake_batches) 


def dimentionality_reduction_plot(real_batches, fake_batches, style_names, model_folder, type="umap"):
    if type == 'umap':
        reduced_points = multi_umap_plot(real_batches, fake_batches)
    elif type == "tsne":
        reduced_points = multi_tsne_plot(real_batches, fake_batches)
    else: 
        raise Exception("No Dimentionality reduction algorthm selected.")
        
    n_styles = len(style_names)

    (n_styles, bs, seq_len, n_sigs) = real_batches.shape

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, n_styles*2))

    plt.figure(figsize=(18, 10))
    for i in range(n_styles):
        ri, ro = i*bs, (i+1)*bs
        fi, fo =  (i+ n_styles) * bs, (i+ n_styles+ 1) * bs
        
        plt.scatter(reduced_points[ri:ro, 0], reduced_points[ri:ro, 1], label=f"Real Style {i+ 1}", alpha=0.5, color=colors[2*i], s=4)
        plt.scatter(reduced_points[fi:fo, 0], reduced_points[fi:fo, 1], label=f"Generated Style {i+ 1}", alpha=0.5, color=colors[2*i+1 ], s=4)
    plt.grid()
    plt.title(f"{type} Reduction of Time Series", fontsize=15)
    plt.ylabel(f"y_{type}", fontsize=15)
    plt.xlabel(f"x_{type}", fontsize=15)
    plt.legend()
    plt.savefig(f"{model_folder}/{type}.png")
    plt.show()

def make_generation_plot(content_sequences, real_style_dset, 
                         ce, se, de,
                         style_names, 
                         model_folder):
    n_style = len(style_names)
    
    fig = plt.figure(figsize=(18, 20))
    spec= fig.add_gridspec(n_style+ 1, 2)

    for label, content_sequence in enumerate(content_sequences):
        
        # plot the content sequence 
        ax_cont_sequence = fig.add_subplot(spec[0, :])
        ax_cont_sequence.set_title(f"Content Sequence label {label}.")
        
        ax_cont_sequence.plot(content_sequence)
        ax_cont_sequence.grid()
        
        for i, style_ in enumerate(style_names):
            real_style_sequence = next(iter(real_style_dset[f"{style_}_valid"]))[0][0]
            fake_sequence = utils.generate(np.array([content_sequence]), np.array([real_style_sequence]), ce, se, de)
        
            _min, _max = np.min(real_style_sequence), np.max(real_style_sequence)
        
            ax = fig.add_subplot(spec[i+1, 0])
            ax.set_title(f"Real Style {i+1}")
            plt.plot(real_style_sequence)
            ax.grid(True)
            ax.set_ylim(_min, _max)
            
            ax = fig.add_subplot(spec[i+1, 1])
            ax.set_title(f"Fake Style {i+1}")
            
            plt.plot(fake_sequence[0])
            ax.set_ylim(_min, _max)
            
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{model_folder}/{label+ 1}_generations.png")


def make_lattent_space_representation(content_sequences, style_datasets, style_names, ce, se, de, model_folder):
    print("[+] Lattent Spcace Representation.")
    
    n_style = len(style_names)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, n_style*2))
    
    for label, content_sequence in enumerate(content_sequences):
        
        fig = plt.figure(figsize=(18, 10))
    
        spec= fig.add_gridspec(2, 2)
            
        ax_content = fig.add_subplot(spec[0, :])
        ax_content_space = fig.add_subplot(spec[1, 0])
        ax_style_space = fig.add_subplot(spec[1, 1])
        
        ax_content.set_title("Content Sequence", fontsize=25)
        ax_content.plot(content_sequence)
        
        ax_content_space.set_title("Content Space", fontsize=25)
        ax_style_space.set_title("Style Space", fontsize=25)
        
        ax_content.grid()
        ax_content_space.grid()
        ax_style_space.grid()
        
        for i, style_ in enumerate(style_names):
            real_style_sequences = next(iter(style_datasets[f"{style_}_valid"]))[0]
            duplicated_contents = np.array([content_sequence]*real_style_sequences.shape[0])
                                    
            real_content_path = ce(duplicated_contents)
            real_style_points = se(real_style_sequences)
            fake_sequences = de([real_content_path, real_style_points])
            
            fake_sequences = tf.concat(fake_sequences, -1)
                        
            fake_content_paths = ce(fake_sequences)
            fake_style_points = se(fake_sequences)
            
            visualization_helpersv2.draw_content_space(ax_content_space, fake_content_paths[0], color=colors[2*i+1], label=f"Gen Seq Style {i}.")
            
            reducer = PCA(n_components=2)
            reducer.fit(real_style_points)
            
            reduced_real_style_points = reducer.transform(real_style_points)
            reduced_fake_style_points = reducer.transform(fake_style_points)
                    
            ax_style_space.scatter(
                reduced_real_style_points[:150, 0], 
                reduced_real_style_points[:150, 1], 
                label=f'Real Style {i}.', 
                alpha=0.25, 
                color=colors[2*i]
            )
        
            ax_style_space.scatter(
                reduced_fake_style_points[:150, 0], 
                reduced_fake_style_points[:150, 1], 
                label=f'Gen Style {i}.', 
                alpha=0.25, 
                color=colors[2*i+ 1]
            )
        
        visualization_helpersv2.draw_content_space(ax_content_space, real_content_path[0], color=colors[0], label=f'Real content sequence')
        ax_content_space.legend(bbox_to_anchor=(-.1, 1.038), loc='upper right')
        ax_style_space.legend(bbox_to_anchor=(1.0, 1.038), loc='upper left')
        
        plt.tight_layout()
        
        filepath = f"{model_folder}/lattent_viz_{label+1}.png"
        plt.savefig(filepath)
        
        print(f"[+] Saved in {filepath}.")    
      



def plot_classif_metric(histories, save_to:str):
    fig = plt.figure(figsize=(18, 10))
    
    loss_axis = plt.subplot(211)
    loss_axis.set_title("Training losses")
    
    acc_axis = plt.subplot(212)
    acc_axis.set_title("Training accuracies")
    
    
    for key, value in histories.items():
        loss_axis.plot(value.history["loss"], ".-", label=f"{key}")
        
        acc_axis.plot(value.history["accuracy"], ".-", label=f"{key}")
    
    loss_axis.grid()
    loss_axis.legend()
    
    acc_axis.grid()
    acc_axis.legend()
    
    plt.tight_layout()
    
    plt.savefig(save_to)
    

def classification_on_lattent_space(real_dataset: dict, encoder:Model, style_names: list, training_parameters: dict, epochs=10):
    # Le but est de savoir si notre content space disposes 
    # Des informations de classes.
    # Nous allons entrainer un model quelconque sur le content space 
    # sur une tache de classification, 
    # Et verifier si le modèle arrive bien à aprendre.
    
    print('[+] Classification on Content Space.')
    
    histories = dict()
    evaluations = dict()
    
    classif_input_shape = encoder.output_shape[1:]
    
    for style in style_names:
        print(f"[+] Train on content space for {style}...")
        dset_train = real_dataset[f"{style}_train"]
        dset_train = dset_train.map(lambda seq, label: (encoder(seq), label))
        
        dset_valid = real_dataset[f"{style}_valid"]
        dset_valid = dset_valid.map(lambda seq, label: (encoder(seq), label))
        
        model = make_naive_discriminator(classif_input_shape, 5)
                
        history = model.fit(dset_train, validation_data=dset_valid, epochs=epochs)
        
        model.evaluate(dset_valid)
        
        histories[style] = history
        evaluations[style] = model.evaluate(dset_valid)[1]
        
    return histories, evaluations

    
def is_content_space_domain_invariant(
    real_content_train:tf.data.Dataset, real_content_valid:tf.data.Dataset,
    real_style_dataset: dict, ce:Model, style_names: list, save_to, epochs=10):
    # Entrainement que sur le content space du content dataset 
    # et on test ses modèles sur les Style datasets.
    
    def plot_learning(hist, save_to: str):
        plt.figure(figsize=(18, 10))
        
        ax= plt.subplot(211)
        ax.plot(hist.history["loss"], '.-', label="loss")
        ax.plot(hist.history["val_loss"], '.-', label="val_loss")
        ax.grid()
        
        ax = plt.subplot(212)
        ax.plot(hist.history["sparse_categorical_accuracy"], '.-', label="accuracy")
        ax.plot(hist.history["val_sparse_categorical_accuracy"], '.-', label="val_accuracy")
        ax.grid()
        
        plt.savefig(save_to)


    accs = {}
    input_shape = ce.output_shape[1:]
    print(input_shape)
    
    model = make_naive_discriminator(input_shape, 5)
    
    cd_train = real_content_train.map(lambda seq, label: (ce(seq), label))
    cd_valid = real_content_valid.map(lambda seq, label: (ce(seq), label))
    
    
    # fit the model on the content space of the content dataset.
    history = model.fit(cd_train, validation_data=cd_valid, epochs=epochs)
    
    plot_learning(history, save_to)
    accs["content_"] = model.evaluate(cd_valid)[1]
    
    for style in style_names:
        print(f"[+] Evaluate on Style {style}")
        style_valid = real_style_dataset[f"{style}_valid"]
        style_valid = style_valid.map(lambda seq, label: (ce(seq), label))
        
        accs[style] = model.evaluate(style_valid)
        
    return history, pd.DataFrame().from_dict(accs)
         
    
    


def main():
    shell_arguments = parse_arguments()
    model_folder = shell_arguments.model_folder
    
    training_params = utils.get_model_training_arguments(model_folder)
    ce, se, de = utils.load_models(model_folder)
    
    # Save model architecture in the folder to have a look.
    # plot_model(ce, f"{model_folder}/content_encoder.png", show_shapes=True)
    # plot_model(se, f"{model_folder}/style_encoder.png", show_shapes=True)
    # plot_model(de, f"{model_folder}/decoder.png", show_shapes=True)
    
    style_names = [get_name(p) for p in training_params["style_datasets"]]
    
    dsets_real, dsets_fake = generate_real_fake_datasets(training_params, ce, se, de)
    
    dset_real_cont_train, dset_real_cont_valid = dataLoader.loading_wrapper(
        training_params["dset_content"], 
        training_params["sequence_lenght_in_sample"], 
        training_params["granularity"],
        training_params["overlap"],
        training_params['batch_size'], drop_labels=False)
    
    dset_real_cont_train = utils.extract_labels(dset_real_cont_train, training_params)
    dset_real_cont_valid = utils.extract_labels(dset_real_cont_valid, training_params)
    
    
    content_sequences = dataLoader.get_seed_visualization_content_sequences(
        training_params["dset_content"], 
        training_params["sequence_lenght_in_sample"])
    
    print("[+] Make Latent Space Representation.")
    make_lattent_space_representation(content_sequences, dsets_real, style_names, ce, se, de, model_folder)
    
    
    print("[+] make generation plot.")
    make_generation_plot(content_sequences, dsets_real, ce, se, de, style_names, model_folder)
    
    
    print(f"[+] Classification on content space.")
    content_classif_hist, content_classif_accs = classification_on_lattent_space(dsets_real, ce, style_names, training_params)
    plot_classif_metric(content_classif_hist, f"{model_folder}/classif_on_content_losses.png")
    pd.DataFrame().from_dict(content_classif_accs).to_excel(f"{model_folder}/evaluation_content_space_classification.xlsx")
    
    
    print(f"[+] Classification on Style space.")
    style_classif_hist, style_classif_accs = classification_on_lattent_space(dsets_real, se, style_names, training_params)
    plot_classif_metric(style_classif_hist, f"{model_folder}/classif_on_style_losses.png")
    pd.DataFrame().from_dict(style_classif_accs).to_excel(f"{model_folder}/evaluation_style_space_classification.xlsx")
    
    content_space_accs = is_content_space_domain_invariant(dset_real_cont_train, dset_real_cont_valid, 
                                                           dsets_real, ce, style_names, f"{model_folder}/domain_invariance_test_learning.png")
    pd.DataFrame().from_dict(content_space_accs).to_excel(f"{model_folder}/evaluation_style_space_classification.xlsx")
    
    
    df_noises, df_ampl, df_time_shift = compute_metrics(dsets_real, dsets_fake, style_names, model_folder)
    
    plot_metric(df_ampl, "Amplitude metric comparison", -0.05, 20.05, f"{model_folder}/amplitude_metric_comparison.png")
    plot_metric(df_noises, "Noise metric comparison", -0.05, 2.505, f"{model_folder}/noise_metric_comparison.png")
    plot_metric(df_time_shift, "time shift metric comparison", -0.05, 25.05, f"{model_folder}/time shift comparison.png")
    
    real_batches, fake_batches = generate_per_style_batch(dsets_real, dsets_fake, style_names)
    
    dimentionality_reduction_plot(real_batches, fake_batches, style_names, model_folder, "umap")
    dimentionality_reduction_plot(real_batches, fake_batches, style_names, model_folder, "tsne")
    
    tstr_stats = tstr_on_styles(dsets_real, dsets_fake, style_names, model_folder)
    


if __name__ == '__main__':
    main()
