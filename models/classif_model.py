from utils import dataLoader
from models.NaiveClassifier import make_naive_discriminator
from utils import utils

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy



class ClassifModel():
    def __init__(self, real_content_dset:str, real_style_dataset_path:list, standard_args:dict, epochs=1):
        
        self.classification_model_folder = "classification_models"
        
        os.makedirs(self.classification_model_folder, exist_ok=True)
    
        sequence_length = standard_args.simulated_arguments.sequence_lenght_in_sample
        gran = standard_args.simulated_arguments.granularity
        overlap = standard_args.simulated_arguments.overlap
        bs = standard_args.simulated_arguments.batch_size
        seq_shape = standard_args.simulated_arguments.seq_shape
        n_classes = standard_args.simulated_arguments.n_classes
        
        _, self.dset_content_valid = dataLoader.loading_wrapper(real_content_dset, sequence_length, gran, overlap, bs, drop_labels=False)
        self.dset_content_valid = self.dset_content_valid.map(lambda seq: (seq[:, :, :-1], seq[:, sequence_length//2, -1]))

        # Load real style datasets.
        self.models = [] # One small classififer per style (I hope.).
        self.valid_set_styles = [] # Save the validation set for later.
        
        for style_path in real_style_dataset_path:
            
            filename = utils.get_name(style_path)
            model_path = f'{self.classification_model_folder}/{filename}.h5'
            
            dset_train, dset_valid = dataLoader.loading_wrapper(style_path, sequence_length, gran, overlap, bs, drop_labels=False)
            
            dset_train = dset_train.map(lambda seq: (seq[:, :, :-1], seq[:, sequence_length//2, -1]))
            dset_valid = dset_valid.map(lambda seq: (seq[:, :, :-1], seq[:, sequence_length//2, -1]))
            
            self.valid_set_styles.append(dset_valid)
            
            print(f"[+] Training model for style {filename}.")
            
            if not os.path.exists(model_path):
                trained_model, history = self.train(dset_train, dset_valid, epochs, seq_shape, n_classes)
                trained_model.save(model_path)
                self.plot_learning_curves(history, f"{self.classification_model_folder}/training_curves/{filename}.png")
                
            else:
                print(f"[+] Loading '{model_path}'")
                trained_model = load_model(model_path)
            
            
            self.models.append(trained_model)
            
            
    def is_already_trained(self, filepath:str):
        return os.path.exists(filepath)
    
    
    def plot_learning_curves(self, history, save_to):
        plt.figure(figsize=(18, 10))    
        
        ax = plt.subplot(211)
        
        plt.plot(history.history["loss"], ".-", label='Train')
        plt.plot(history.history["val_loss"], ".-", label='Valid')
        
        ax.grid(True)
        ax.legend()
        
        ax = plt.subplot(212)
        
        plt.plot(history.history["sparse_categorical_accuracy"], ".-", label='Train')
        plt.plot(history.history["val_sparse_categorical_accuracy"], ".-", label='Valid')
        
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend()
        
        plt.savefig(save_to)

    
    def train(self, train_dset, valid_dset, epochs, seq_shape, n_classes):
        model = make_naive_discriminator(seq_shape, n_classes)
        history = model.fit(train_dset, validation_data=valid_dset, epochs=epochs)
        return model, history
    
    def generate(self, ce:Model, se:Model, de:Model, cont_batch, style_batch):
        content_encodings = ce(cont_batch)
        style_encodings = se(style_batch)
        sequences = de([content_encodings, style_encodings])
        return tf.concat(sequences, -1)
    
    
    def evaluate(self, content_encoder:Model, style_encoder:Model, decoder:Model):
        print('[+] Classitifation metric.')
        accs = []
        
        acc_metric = SparseCategoricalAccuracy()
        
        for i in range(len(self.valid_set_styles)):
            
            selected_valid_style = self.valid_set_styles[i]
            selected_model = self.models[i]
            acc_metric.reset_state()
            
            for (content_batch, content_label), (style_batch, _) in zip(self.dset_content_valid.take(100), selected_valid_style.take(100)):
                generated_batch = self.generate(content_encoder, style_encoder, decoder, content_batch, style_batch)
                
                model_pred_on_synth = selected_model(generated_batch)
                
                acc_metric.update_state(content_label, model_pred_on_synth)
                print(f"\r Style {i}; {acc_metric.result().numpy():0.3f}", end="")
                
            accs.append(acc_metric.result().numpy())
            
        return np.mean(accs)
       