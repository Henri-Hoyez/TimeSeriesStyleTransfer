from models.mtsStyleTransferV1 import ContentEncoder, Decoder, GlobalDiscriminator, LocalDiscriminator, StyleEncoder
from utils.dataLoader import get_batches
from utils.tensorboard_log import TensorboardLog
from utils import metric, simple_metric, visualization_helpersv2, MLFlow_utils
from models.classif_model import ClassifModel


from models import losses
import numpy as np
from sklearn.decomposition import PCA

from tensorflow.keras.metrics import BinaryAccuracy, binary_accuracy
from tensorflow.keras.optimizers import RMSprop, Adam
# from tensorflow.keras.utils import plot_model

from keras._tf_keras.keras.utils import plot_model

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf
from datetime import datetime

class Trainer():
    def __init__(self, shell_arguments, default_arguments, ) -> None:
        
        self.shell_arguments = shell_arguments
        self.default_arguments = default_arguments
        n_styles = len(shell_arguments.style_datasets)

        sequence_length = default_arguments.simulated_arguments.sequence_lenght_in_sample
        n_signals = default_arguments.simulated_arguments.n_feature
        feat_wiener = default_arguments.simulated_arguments.n_wiener
        n_sample_wiener = default_arguments.simulated_arguments.n_sample_wiener
        style_vector_size = default_arguments.simulated_arguments.style_vector_size

        self.batch_size = self.default_arguments.simulated_arguments.batch_size
        self.epochs = self.default_arguments.simulated_arguments.epochs

        self.l_triplet = self.default_arguments.simulated_arguments.l_triplet
        self.l_disentanglement = self.default_arguments.simulated_arguments.l_disentanglement
        self.l_reconstr = self.default_arguments.simulated_arguments.l_reconstr
        self.l_global = self.default_arguments.simulated_arguments.l_global
        self.style_preservation = self.default_arguments.simulated_arguments.l_style_preservation
        self.l_local = self.default_arguments.simulated_arguments.l_local
        self.l_content = self.default_arguments.simulated_arguments.l_content
        self.e_adv = self.default_arguments.simulated_arguments.encoder_adv

        # self.style_encoder_adv = self.default_arguments.simulated_arguments.style_encoder_adv
        self.discr_success_th = self.default_arguments.simulated_arguments.discriminator_success_threashold
        self.normal_training_epochs = self.default_arguments.simulated_arguments.normal_training_epochs

        self.global_discr_acc_train = BinaryAccuracy()
        self.global_discr_acc_valid = BinaryAccuracy()

        self.local_discr_acc_train = BinaryAccuracy()
        self.local_discr_acc_valid = BinaryAccuracy()

        self.content_encoder = ContentEncoder.make_content_encoder(sequence_length, n_signals, feat_wiener)
        self.style_encoder = StyleEncoder.make_style_encoder(sequence_length, n_signals, style_vector_size)
        self.decoder = Decoder.make_generator(n_sample_wiener, feat_wiener, style_vector_size ,n_signals)
        self.global_discriminator = GlobalDiscriminator.make_global_discriminator(sequence_length, n_signals, n_styles)
        self.local_discriminator = LocalDiscriminator.create_local_discriminator(n_signals, sequence_length, n_styles)
        
        # Prepare the classification metric.
        self.classif_metric = ClassifModel(shell_arguments.content_dset, shell_arguments.style_datasets, default_arguments)
        
        self.prepare()
        self.prepare_loggers(n_styles)


    def plot_models(self):

        plot_model(self.decoder, show_shapes=True, to_file='Decoder.png',  expand_nested=True)
        plot_model(self.global_discriminator, show_shapes=True, to_file='global_discriminator.png',  expand_nested=True)
        plot_model(self.local_discriminator, show_shapes=True, to_file='local_discriminator.png',  expand_nested=True)
        plot_model(self.content_encoder, show_shapes=True, to_file='content_encoder.png',  expand_nested=True)
        plot_model(self.style_encoder, show_shapes=True, to_file='style_encoder.png',  expand_nested=True)


    def generate(self, content_batch, style_batch):
        content = self.content_encoder(content_batch, training=False)
        style = self.style_encoder(style_batch, training=False)
        generated = self.decoder([content, style], training=False)
        generated = tf.concat(generated, -1)
        return generated
    
    
    def train(self):
        total_batch_train = "?"
        total_batch_valid = "?"
        
        gd_sucess = self.discr_success_th
        ld_sucess = self.discr_success_th

        discr_success = self.discr_success_th
        discr_success_valid = self.discr_success_th
        alpha = self.default_arguments.simulated_arguments.alpha

        for e in range(self.default_arguments.simulated_arguments.epochs):
            self.logger.reset_metric_states()
            self.logger.reset_valid_states()

            self.global_discr_acc_train.reset_state()
            self.global_discr_acc_valid.reset_state()

            self.local_discr_acc_train.reset_state()
            self.local_discr_acc_valid.reset_state()
            
            print("[+] Train Step...")  
            for i, (content_batch, style_batch) in enumerate(zip(self.dset_content_train, self.dsets_style_train)):

                content_sequence1 = content_batch[:int(self.batch_size)]
                content_sequence2 = content_batch[int(self.batch_size):]
    
                # Then train one another given their performances. 
                force_backward = i == 0 and e == 0                         # Force the full execution for tensorflow mapping.  
                train_global_d = bool(gd_sucess < self.discr_success_th) or True
                train_local_d = bool(ld_sucess < self.discr_success_th) or True
                train_g = (not train_global_d and not train_local_d) or True
            
            
                if train_global_d:
                    test = 'Train Global Discriminator'
                    
                if train_local_d:
                    test = "Train Local Discriminator      "
                    
                if train_g:
                    test = "Train Generator            "

                global_d_accs, local_d_accs = self.discriminator_step(content_sequence1, style_batch, train_global_d or force_backward, train_local_d or force_backward)

                self.generator_step(content_sequence1, content_sequence2, style_batch, train_g or force_backward)
                
                gd_sucess = gd_sucess * (1. - alpha) + alpha * (global_d_accs)      
                ld_sucess = ld_sucess * (1. - alpha) + alpha * (local_d_accs)  
                
                discr_accs =  np.max([global_d_accs, local_d_accs])
                discr_success = discr_success * (1. - alpha) + alpha * (discr_accs)

                self.logger.print_train(e, self.epochs, i, total_batch_train, f"[{global_d_accs:0.2f}; {local_d_accs:0.2f}]: {discr_accs:0.4f} -> {discr_success:0.4f} {test}")
            
            print()
            print("[+] Validation Step...")
            for vb, (content_batch, style_batch) in enumerate(zip(self.dset_content_valid, self.dsets_style_valid)):
                content_sequence1 = content_batch[:int(self.batch_size)]
                content_sequence2 = content_batch[int(self.batch_size):]

                global_d_accs, local_d_accs = self.discriminator_valid(content_sequence1, style_batch)

                # discr_accs =  central_channel_d_treadoff* global_d_accs + (1- central_channel_d_treadoff)* local_d_accs
                discr_accs =  np.max([global_d_accs, local_d_accs])

                discr_success_valid = discr_success_valid * (1. - alpha) + alpha * (discr_accs)

                self.generator_valid(content_sequence1, content_sequence2, style_batch)

                self.logger.print_valid(e, self.epochs, vb, total_batch_valid, f"[{global_d_accs:0.2f}; {local_d_accs:0.2f}]: {discr_accs:0.4f} -> {discr_success_valid:0.4f}")

            self.training_evaluation(e)
            self.logger.log_train_value("40 - Discriminator Sucess", discr_success, e)            
            self.logger.log_valid_value("40 - Discriminator Sucess", discr_success_valid, e)     
            
            

            if e == 0:
                total_batch_train = i
                total_batch_valid = vb
        
        self.save()
        
    def multistyle_viz(self, epoch:int):
        
        for i, content_sequence in enumerate(self.content_viz_sequences):
        
            save_to = f"{self.logger.full_path}/{epoch}_{i}.png"  
        
            # Generate Sequences with the same content and all styles.
            content_of_content = self.content_encoder(np.array([content_sequence]))
            content_for_generation = np.array([content_of_content] * self.seed_styles_valid.shape[1])
            shape = content_for_generation.shape
            content_for_generation = content_for_generation.reshape((-1, shape[-2], shape[-1]))
            
            # Make the Style Space for the Real and Simulated Sequences.
            real_style_space = np.array([ self.style_encoder(style_sequences) for style_sequences in self.seed_styles_valid ])
            
            # Generate sequence of different style, but with the same content.
            generated_sequences = np.array([ tf.concat(self.decoder([content_for_generation, style_vectors]), -1) for style_vectors in real_style_space ])
            
            # Extract the content.
            content_of_gens = np.array([ self.content_encoder(generated_sequence) for generated_sequence in generated_sequences ])
            
            # Extract the style.
            style_of_gen = np.array([ self.style_encoder(generated_sequence) for generated_sequence in generated_sequences ])
            
            # Reduce the dimentionality for the style.
            pca = PCA(2)
                    
            all_styles = tf.concat((real_style_space, style_of_gen), 0)
            all_styles = tf.reshape(all_styles, (-1, all_styles.shape[-1]))
                    
            pca = pca.fit(all_styles)
            
            real_reduced_styles = np.array([ pca.transform(particular_style_space) for particular_style_space in real_style_space ])
            gen_reduced_styles = np.array([ pca.transform(particular_style_space) for particular_style_space in style_of_gen ])
                    
            visualization_helpersv2.plot_multistyle_sequences(
                content_sequence, 
                self.seed_styles_valid[:, 0], 
                generated_sequences[:, 0], 
                content_of_content, content_of_gens[:, 0],
                real_reduced_styles, gen_reduced_styles,
                epoch, save_to
                )

    
    def training_evaluation(self, epoch):

        mean_acc = self.classif_metric.evaluate(self.content_encoder, self.style_encoder, self.decoder)
        self.logger.valid_loggers["00 - Model Classification Acc"](mean_acc)
        
        generation_style_train = np.array([self.generate(self.seed_content_train, style_train) for style_train in self.seed_styles_train])
        generation_style_valid = np.array([self.generate(self.seed_content_valid, style_train) for style_train in self.seed_styles_valid])

        # self.metric_evaluation(generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid)
        
        self.simple_noise_metric(generation_style_train, generation_style_valid, epoch)

        self.simple_amplitude_metric(generation_style_train, generation_style_valid, epoch)

        plot_buff = self.make_viz()
        
        # Make multistyle visualization. 
        self.multistyle_viz(epoch)
        
        self.logger.log_train(plot_buff, epoch)
        self.logger.log_valid(epoch)
            

    # def metric_evaluation(self, generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid):
    #     metric_s1_train = metric.compute_metric(generation_style1_train, self.seed_styles_train[0], self.default_arguments.simulated_arguments)
    #     metric_s1_valid = metric.compute_metric(generation_style1_valid, self.seed_styles_valid[0], self.default_arguments.simulated_arguments)

    #     metric_s2_train = metric.compute_metric(generation_style2_train, self.seed_styles_train[1], self.default_arguments.simulated_arguments)
    #     metric_s2_valid = metric.compute_metric(generation_style2_valid, self.seed_styles_valid[1], self.default_arguments.simulated_arguments)
        
    #     self.logger.met_corr_style1_train(metric_s1_train)
    #     self.logger.met_corr_style2_train(metric_s2_train)

    #     self.logger.met_corr_style1_valid(metric_s1_valid)
    #     self.logger.met_corr_style2_valid(metric_s2_valid)

    def simple_noise_metric(self, generation_style_train, generation_style_valid, epoch):
        
        placeholder = "00 - Noise Similarity Style"
        
        seed_content_trends_train, _ = simple_metric.simple_metric_on_noise(self.seed_content_train)
        seed_content_trends_valid, _ = simple_metric.simple_metric_on_noise(self.seed_content_valid)
        
        for i in range(generation_style_train.shape[0]):
            noise_key = f"{placeholder} {i+ 1}"
            content_key = f"02 - Content Similarity Style {i+ 1}"
            
            content_gen_train, generated_noise_train = simple_metric.simple_metric_on_noise(generation_style_train[i])
            content_gen_valid, generated_noise_valid = simple_metric.simple_metric_on_noise(generation_style_valid[i])
            
            _, seed_style_noise_train = simple_metric.simple_metric_on_noise(self.seed_styles_train[i])
            _, seed_style_noise_valid = simple_metric.simple_metric_on_noise(self.seed_styles_valid[i])
            
            noise_similarity_train = np.mean(np.abs(seed_style_noise_train - generated_noise_train))
            noise_similarity_valid = np.mean(np.abs(seed_style_noise_valid - generated_noise_valid))
            
            content_similarity_train = np.mean(np.abs(seed_content_trends_train - content_gen_train))
            content_similarity_valid = np.mean(np.abs(seed_content_trends_valid - content_gen_valid))
            
            self.logger.train_loggers[noise_key](noise_similarity_train)
            self.logger.valid_loggers[noise_key](noise_similarity_valid)
            
            self.logger.train_loggers[content_key](content_similarity_train)
            self.logger.valid_loggers[content_key](content_similarity_valid)
            
        train_mean_noise = self.logger.get_mean_metric(self.logger.train_loggers, 'Noise')
        valid_mean_noise = self.logger.get_mean_metric(self.logger.valid_loggers, 'Noise')
        
        self.logger.log_train_value(f"{placeholder} Mean", train_mean_noise, epoch)
        self.logger.log_valid_value(f"{placeholder} Mean", valid_mean_noise, epoch)
    

    def simple_amplitude_metric(self, generation_style_train, generation_style_valid, epoch):
        
        placeholder = f'01 - Amplitude Similarity Style'
        
        for i in range(generation_style_train.shape[0]):
            ampli_key = f"{placeholder} {i+ 1}"
            
            ampl_diff_train = simple_metric.simple_amplitude_metric(self.seed_styles_train[i], generation_style_train[i])
            ampl_diff_valid = simple_metric.simple_amplitude_metric(self.seed_styles_valid[i], generation_style_valid[i])

            self.logger.train_loggers[ampli_key](ampl_diff_train)
            self.logger.valid_loggers[ampli_key](ampl_diff_valid)
            
        train_mean_ampl = self.logger.get_mean_metric(self.logger.train_loggers, 'Amplitude')
        valid_mean_ampl = self.logger.get_mean_metric(self.logger.valid_loggers, 'Amplitude')
        
        self.logger.log_train_value(f'{placeholder} Mean', train_mean_ampl, epoch)
        self.logger.log_valid_value(f'{placeholder} Mean', valid_mean_ampl, epoch)
            
        

    def make_viz(self):
        
        vis_fig = visualization_helpersv2.plot_generated_sequence(
            self.content_encoder, self.style_encoder, self.decoder, 
            self.seed_content_valid, self.seed_styles_valid
        )
        plot_buff = self.logger.fig_to_buff(vis_fig)

        return plot_buff

    def save(self):
        root = f"{self.shell_arguments.save_to}/{self.shell_arguments.exp_folder}/{self.shell_arguments.exp_name}"
        if not os.path.exists(root):
            print(f"[!] Save Folder Missing... Create root save folder at {root}")
            os.makedirs(root)

        print(f"[+] Saving to {root}")
        self.content_encoder.save(f"{root}/content_encoder.h5")
        self.style_encoder.save(f"{root}/style_encoder.h5")
        self.decoder.save(f"{root}/decoder.h5")
        self.global_discriminator.save(f"{root}/global_discriminator.h5")
        self.local_discriminator.save(f"{root}/local_discriminator.h5")
        print("[+] Save Parameters...")

        parameters = {
            "style_datasets":self.shell_arguments.style_datasets, 
            "dset_content":self.shell_arguments.content_dset,
            "sequence_lenght_in_sample":self.default_arguments.simulated_arguments.sequence_lenght_in_sample,
            "granularity":self.default_arguments.simulated_arguments.granularity,
            "overlap":self.default_arguments.simulated_arguments.overlap,
            "epochs":self.default_arguments.simulated_arguments.epochs,
            "n_feature":self.default_arguments.simulated_arguments.n_feature,
            "seq_shape":self.default_arguments.simulated_arguments.seq_shape,
            "n_classes":self.default_arguments.simulated_arguments.n_classes,
            "batch_size":self.default_arguments.simulated_arguments.batch_size,
            "n_styles":self.default_arguments.simulated_arguments.n_styles,
            "style_vector_size":self.default_arguments.simulated_arguments.style_vector_size,
            "n_wiener":self.default_arguments.simulated_arguments.n_wiener,
            "n_sample_wiener":self.default_arguments.simulated_arguments.n_sample_wiener,
            "noise_dim":self.default_arguments.simulated_arguments.noise_dim,
            "n_validation_sequences":self.default_arguments.simulated_arguments.n_validation_sequences,
            "discrinator_step":self.default_arguments.simulated_arguments.discrinator_step,
            "l_reconstr":self.default_arguments.simulated_arguments.l_reconstr,
            "l_local":self.default_arguments.simulated_arguments.l_local,
            "l_global":self.default_arguments.simulated_arguments.l_global,
            "l_style_preservation":self.default_arguments.simulated_arguments.l_style_preservation,
            "l_content":self.default_arguments.simulated_arguments.l_content,
            "encoder_adv":self.default_arguments.simulated_arguments.encoder_adv,
            "l_disentanglement":self.default_arguments.simulated_arguments.l_disentanglement,
            "l_triplet":self.default_arguments.simulated_arguments.l_triplet,
            "triplet_r":self.default_arguments.simulated_arguments.triplet_r,
            "discriminator_success_threashold":self.default_arguments.simulated_arguments.discriminator_success_threashold,
            "alpha":self.default_arguments.simulated_arguments.alpha,
            "normal_training_epochs":self.default_arguments.simulated_arguments.normal_training_epochs,
            "note":self.default_arguments.note
        }

        MLFlow_utils.save_configuration(f"{root}/model_config.json", parameters)
        print("[+] Saved !")

    def set_seeds(self, _seed_style_train, _seed_style_valid, content_viz_sequences):

        self.seed_styles_train = _seed_style_train
        self.seed_styles_valid = _seed_style_valid
        
        self.seed_content_train = get_batches(self.dset_content_train, 1)
        self.seed_content_valid = get_batches(self.dset_content_valid, 1)
        
        self.content_viz_sequences = content_viz_sequences


    def prepare_loggers(self, n_style:int):
        
        noise_metric = []
        ampli_metric = []
        content_sim_metric = []
        
        for i in range(n_style):
            noise_metric.append(f"00 - Noise Similarity Style {i+ 1}")    
            ampli_metric.append(f"01 - Amplitude Similarity Style {i+ 1}")
            content_sim_metric.append(f"02 - Content Similarity Style {i+ 1}")
            
            
        metric_keys = [
            
            
            "00 - Model Classification Acc",
            "10 - Total Generator Loss", 
            "11 - Reconstruction from Content",
            "12 - Central Realness",
            "13 - Local Realness",
            
            "20 - Style Loss",
            "21 - Triplet Loss",
            "22 - Disentanglement Loss",
            
            "30 - Content Loss",
            
            "40 - Global Discriminator Loss",
            "40 - Global Discriminator Acc", 
            
            "40 - Local Discriminator Loss",
            "40 - Local Discriminator Acc", 
            
            "41 - Global Discriminator Style Loss (Real Data)",
            "41 - Global Discriminator Style Loss (Fake Data)",
            
            "42 - Local Discriminator Style Loss (Real Data)",
            "42 - Local Discriminator Style Loss (Fake Data)",
            ]
        
    
        metric_keys.extend(noise_metric)
        metric_keys.extend(ampli_metric)
        metric_keys.extend(content_sim_metric)
        
        self.logger = TensorboardLog(self.shell_arguments, metric_keys)

    def prepare(self):
        self.opt_content_encoder = RMSprop(learning_rate=0.001) # 0.0005
        self.opt_style_encoder = RMSprop(learning_rate=0.001) # 0.0005
        self.opt_decoder = RMSprop(learning_rate=0.001) # 0.0005
        self.local_discriminator_opt = RMSprop(learning_rate=0.001) # 0.0005
        self.global_discriminator_opt = RMSprop(learning_rate=0.001) # 0.0005 

    def instanciate_datasets(self, 
                             content_dset_train:tf.data.Dataset, 
                             content_dset_valid:tf.data.Dataset, 
                             styles_dset_train:tf.data.Dataset, 
                             styles_dset_valid:tf.data.Dataset):
        
        self.dset_content_train = content_dset_train
        self.dset_content_valid = content_dset_valid

        self.dsets_style_train = styles_dset_train
        self.dsets_style_valid = styles_dset_valid


    @tf.function
    def discriminator_step(self, content_sequence1, style_batch, g_backward, l_backward):
        # Discriminator Step

        style_sequences, style_labels = style_batch[0], style_batch[1]

        with tf.GradientTape(persistent=True) as discr_tape:
            # Sequence generations.
            c1 = self.content_encoder(content_sequence1, training=False)

            style_encoded = self.style_encoder(style_sequences, training=False)

            generated= self.decoder([c1, style_encoded], training=False)

            # split sequences for discriminator's inputs
            real_style_sequences_splitted = tf.split(style_sequences, style_sequences.shape[-1], axis=-1)

            # Global on Real
            g_crit_real, g_style_classif_real = self.global_discriminator(real_style_sequences_splitted, training=True)

            # Local on Real
            l_crit_real = self.local_discriminator(real_style_sequences_splitted, training=True)

            # Global on Generated
            g_crit_fake, _ = self.global_discriminator(generated, training=True)

            # Local on fake
            l_crit_fake = self.local_discriminator(generated, training=True)

            # Compute the loss for GLOBAL the Discriminator
            # g_crit_loss = losses.discriminator_loss(g_crit_real, g_crit_fake)
            g_crit_loss = losses.least_square_discriminator_loss(g_crit_real, g_crit_fake)
            g_style_real = losses.style_classsification_loss(g_style_classif_real, style_labels)

            # l_loss = losses.local_discriminator_loss(l_crit_real, l_crit_fake)
            l_loss = losses.least_square_local_discriminator_loss(l_crit_real, l_crit_fake)

        # (GOBAL DISCRIMINATOR): Real / Fake and style
        global_discr_gradient = discr_tape.gradient([g_crit_loss, g_style_real], self.global_discriminator.trainable_variables)
        grads = discr_tape.gradient(l_loss, self.local_discriminator.trainable_variables)

        if g_backward:
            self.global_discriminator_opt.apply_gradients(zip(global_discr_gradient, self.global_discriminator.trainable_variables)) 
            
        if l_backward:
            self.local_discriminator_opt.apply_gradients(zip(grads, self.local_discriminator.trainable_variables))
    

        # Calculate the performances of the Discriminator.
        real_labels = tf.ones_like(g_crit_real)
        generation_labels = tf.zeros_like(g_crit_fake)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.

        global_accs = tf.reduce_mean([
            binary_accuracy(real_labels, g_crit_real),
            binary_accuracy(generation_labels, g_crit_fake)
        ])

        channel_accs = tf.reduce_mean([
            losses.local_discriminator_accuracy(real_labels, l_crit_real),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake)
        ])
        
        self.logger.train_loggers['40 - Global Discriminator Loss'](g_crit_loss)
        self.logger.train_loggers['41 - Global Discriminator Style Loss (Real Data)'](g_style_real)
        self.logger.train_loggers['40 - Local Discriminator Loss'](l_loss)

        self.logger.train_loggers['40 - Global Discriminator Acc'](global_accs)
        self.logger.train_loggers["40 - Local Discriminator Acc"](channel_accs)

        return global_accs, channel_accs


    @tf.function
    def generator_step(self, content_sequence1, content_sequence2, style_batch, backward=True):

        style_sequences, style_labels = style_batch[0],  style_batch[1]

        # Here, things get a little bit more complicated :)
        with tf.GradientTape() as content_tape, tf.GradientTape() as style_tape, tf.GradientTape() as decoder_tape:
            # Reconstruction Loss: Try to generate the same sequence given
            # it's content and style.
            contents = tf.concat([content_sequence1, content_sequence2], 0)
            cs = self.content_encoder(contents, training=True)
            s_cs = self.style_encoder(contents, training=True)
            id_generated = self.decoder([cs, s_cs], training=True)
            id_generated = tf.concat(id_generated, -1)

            reconstr_loss = losses.recontruction_loss(contents, id_generated)

            ####
            styles = tf.concat([style_sequences, style_sequences], 0)   
            style_label_extended = tf.concat([style_labels, style_labels], 0)   
            _bs = content_sequence1.shape[0]

            encoded_content= self.content_encoder(contents, training=True)
            encoded_styles = self.style_encoder(styles, training=True)

            generations = self.decoder([encoded_content, encoded_styles], training=True)

            merged_generations = tf.concat(generations, -1)

            s_generations = self.style_encoder(merged_generations, training=True)
            c_generations = self.content_encoder(merged_generations, training=True)

            # Discriminator pass for the adversarial loss for the generator.

            crit_on_fake, style_classif_fakes = self.global_discriminator(generations, training=False)

            # Local Discriminator on Fake Data.
            l_crit_on_fake = self.local_discriminator(generations, training=False)

            # Channel Discriminator losses
            # local_realness_loss = losses.local_generator_loss(l_crit_on_fake)
            local_realness_loss = losses.least_square_local_generator_loss(l_crit_on_fake)
            
            # Global Generator losses.

            global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_label_extended)
            # global_realness_loss = losses.generator_loss(crit_on_fake)
            global_realness_loss = losses.least_square_generator_loss(crit_on_fake)

            ########
            content_preservation = losses.fixed_point_content(encoded_content, c_generations)

            s_c1_s = s_generations[:_bs]
            s_c2_s = s_generations[_bs:]

            content_style_disentenglement = losses.fixed_point_disentanglement(s_c2_s, s_c1_s, encoded_styles[:_bs])

            # triplet_style =  losses.get_triplet_loss(anchor, positive, negative, self.default_arguments.simulated_arguments.triplet_r)
            triplet_style1 = losses.hard_triplet(style_labels, s_c1_s, self.default_arguments.simulated_arguments.triplet_r)
            triplet_style2 = losses.hard_triplet(style_labels, s_c2_s, self.default_arguments.simulated_arguments.triplet_r)
            triplet_style = (triplet_style1+ triplet_style2)/2

            content_encoder_loss = self.l_content* content_preservation+ self.e_adv* global_realness_loss + self.e_adv* global_style_loss
            style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement  + self.e_adv* global_realness_loss + self.e_adv* global_style_loss

            g_loss = self.l_reconstr* reconstr_loss+ self.l_global* global_realness_loss + self.style_preservation* global_style_loss+ self.l_local* local_realness_loss

        # Make the Networks Learn!
        content_grad=content_tape.gradient(content_encoder_loss, self.content_encoder.trainable_variables)
        style_grad = style_tape.gradient(style_encoder_loss, self.style_encoder.trainable_variables)
        decoder_grad = decoder_tape.gradient(g_loss, self.decoder.trainable_variables)
            
        if backward == True:
            self.opt_decoder.apply_gradients(zip(decoder_grad, self.decoder.trainable_variables))

        self.opt_content_encoder.apply_gradients(zip(content_grad, self.content_encoder.trainable_variables))
        self.opt_style_encoder.apply_gradients(zip(style_grad, self.style_encoder.trainable_variables))

        self.logger.train_loggers['10 - Total Generator Loss'](g_loss)
        self.logger.train_loggers["11 - Reconstruction from Content"](reconstr_loss)

        self.logger.train_loggers["13 - Local Realness"](local_realness_loss)
        self.logger.train_loggers["12 - Central Realness"](global_realness_loss)

        self.logger.train_loggers["41 - Global Discriminator Style Loss (Fake Data)"](global_style_loss)

        self.logger.train_loggers["22 - Disentanglement Loss"](content_style_disentenglement)
        self.logger.train_loggers["21 - Triplet Loss"](triplet_style)
        self.logger.train_loggers["20 - Style Loss"](style_encoder_loss)
        self.logger.train_loggers["30 - Content Loss"](content_preservation)

    @tf.function
    def generator_valid(self, content_sequence1, content_sequence2, style_batch):
        
        style_sequences, style_labels = style_batch[0],  style_batch[1]
        
        # Reconstruction Loss: Try to generate the same sequence given
        # it's content and style.
        contents = tf.concat([content_sequence1, content_sequence2], 0)
        cs = self.content_encoder(contents, training=True)
        s_cs = self.style_encoder(contents, training=True)
        id_generated = self.decoder([cs, s_cs], training=True)
        id_generated = tf.concat(id_generated, -1)

        reconstr_loss = losses.recontruction_loss(contents, id_generated)

        ####
        styles = tf.concat([style_sequences, style_sequences], 0)   
        style_label_extended = tf.concat([style_labels, style_labels], 0)   
        _bs = content_sequence1.shape[0]

        encoded_content= self.content_encoder(contents, training=True)
        encoded_styles = self.style_encoder(styles, training=True)

        generations = self.decoder([encoded_content, encoded_styles], training=True)

        merged_generations = tf.concat(generations, -1)

        s_generations = self.style_encoder(merged_generations, training=True)
        c_generations = self.content_encoder(merged_generations, training=True)

        # Discriminator pass for the adversarial loss for the generator.

        crit_on_fake, style_classif_fakes = self.global_discriminator(generations, training=False)

        # Local Discriminator on Fake Data.
        l_crit_on_fake = self.local_discriminator(generations, training=False)

        # Channel Discriminator losses
        # local_realness_loss = losses.local_generator_loss(l_crit_on_fake)
        local_realness_loss = losses.least_square_local_generator_loss(l_crit_on_fake)
        
        # Global Generator losses.

        global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_label_extended)
        # global_realness_loss = losses.generator_loss(crit_on_fake)
        global_realness_loss = losses.least_square_generator_loss(crit_on_fake)

        ########
        content_preservation = losses.fixed_point_content(encoded_content, c_generations)

        s_c1_s = s_generations[:_bs]
        s_c2_s = s_generations[_bs:]

        content_style_disentenglement = losses.fixed_point_disentanglement(s_c2_s, s_c1_s, encoded_styles[:_bs])

        # triplet_style =  losses.get_triplet_loss(anchor, positive, negative, self.default_arguments.simulated_arguments.triplet_r)
        triplet_style1 = losses.hard_triplet(style_labels, s_c1_s, self.default_arguments.simulated_arguments.triplet_r)
        triplet_style2 = losses.hard_triplet(style_labels, s_c2_s, self.default_arguments.simulated_arguments.triplet_r)
        triplet_style = (triplet_style1+ triplet_style2)/2

        content_encoder_loss = self.l_content* content_preservation+ self.e_adv* global_realness_loss + self.e_adv* global_style_loss
        style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement  + self.e_adv* global_realness_loss + self.e_adv* global_style_loss

        g_loss = self.l_reconstr* reconstr_loss+ self.l_global* global_realness_loss + self.style_preservation* global_style_loss+ self.l_local* local_realness_loss

        self.logger.valid_loggers['10 - Total Generator Loss'](g_loss)
        self.logger.valid_loggers["11 - Reconstruction from Content"](reconstr_loss)

        self.logger.valid_loggers["13 - Local Realness"](local_realness_loss)
        self.logger.valid_loggers["12 - Central Realness"](global_realness_loss)

        self.logger.valid_loggers["41 - Global Discriminator Style Loss (Fake Data)"](global_style_loss)

        self.logger.valid_loggers["21 - Triplet Loss"](triplet_style)
        self.logger.valid_loggers["20 - Style Loss"](style_encoder_loss)
        self.logger.valid_loggers["30 - Content Loss"](content_preservation)
        self.logger.valid_loggers["22 - Disentanglement Loss"](content_style_disentenglement)



    @tf.function
    def discriminator_valid(self, content_sequence1, style_batch):
        style_sequences, style_labels = style_batch[0], style_batch[1]

        # Sequence generations.
        c1 = self.content_encoder(content_sequence1, training=False)

        style_encoded = self.style_encoder(style_sequences, training=False)

        generated= self.decoder([c1, style_encoded], training=False)

        # split sequences for discriminator's inputs
        real_style_sequences_splitted = tf.split(style_sequences, style_sequences.shape[-1], axis=-1)

        # Global on Real
        g_crit_real, g_style_classif_real = self.global_discriminator(real_style_sequences_splitted, training=True)

        # Local on Real
        l_crit_real = self.local_discriminator(real_style_sequences_splitted, training=True)

        # Global on Generated
        g_crit_fake, _ = self.global_discriminator(generated, training=True)

        # Local on fake
        l_crit_fake = self.local_discriminator(generated, training=True)

        # Compute the loss for GLOBAL the Discriminator
        # g_crit_loss = losses.discriminator_loss(g_crit_real, g_crit_fake)
        g_crit_loss = losses.least_square_discriminator_loss(g_crit_real, g_crit_fake)
        g_style_real = losses.style_classsification_loss(g_style_classif_real, style_labels)

        # l_loss = losses.local_discriminator_loss(l_crit_real, l_crit_fake)
        l_loss = losses.least_square_discriminator_loss(l_crit_real, l_crit_fake)

        # Calculate the performances of the Discriminator.
        real_labels = tf.ones_like(g_crit_real)
        generation_labels = tf.zeros_like(g_crit_fake)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.
        global_accs = tf.reduce_mean([
            binary_accuracy(real_labels, g_crit_real),
            binary_accuracy(generation_labels, g_crit_fake)
        ])

        channel_accs = tf.reduce_mean([
            losses.local_discriminator_accuracy(real_labels, l_crit_real),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake)
        ])
        
        self.logger.valid_loggers['40 - Global Discriminator Loss'](g_crit_loss)
        self.logger.valid_loggers['41 - Global Discriminator Style Loss (Real Data)'](g_style_real)
        self.logger.valid_loggers['40 - Local Discriminator Loss'](l_loss)

        self.logger.valid_loggers['40 - Global Discriminator Acc'](global_accs)
        self.logger.valid_loggers["40 - Local Discriminator Acc"](channel_accs)

        return global_accs, channel_accs
    
