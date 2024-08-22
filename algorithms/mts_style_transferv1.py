from models.mtsStyleTransferV1 import ContentEncoder, Decoder, GlobalDiscriminator, LocalDiscriminator, StyleEncoder
from utils.dataLoader import get_batches
from utils.tensorboard_log import TensorboardLog
from utils import metric, simple_metric, visualization_helpers, MLFlow_utils

from models import losses
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf


class Trainer():
    def __init__(self, shell_arguments, default_arguments) -> None:
        self.shell_arguments = shell_arguments
        self.default_arguments = default_arguments

        sequence_length = default_arguments.simulated_arguments.sequence_lenght_in_sample
        n_signals = default_arguments.simulated_arguments.n_feature
        feat_wiener = default_arguments.simulated_arguments.n_wiener
        n_sample_wiener = default_arguments.simulated_arguments.n_sample_wiener
        style_vector_size = default_arguments.simulated_arguments.style_vector_size
        n_styles = default_arguments.simulated_arguments.n_styles

        self.batch_size = self.default_arguments.simulated_arguments.batch_size
        self.discr_step = self.default_arguments.simulated_arguments.discrinator_step
        self.epochs = self.default_arguments.simulated_arguments.epochs

        self.l_triplet = self.default_arguments.simulated_arguments.l_triplet
        self.l_disentanglement = self.default_arguments.simulated_arguments.l_disentanglement
        self.l_reconstr = self.default_arguments.simulated_arguments.l_reconstr
        self.l_global = self.default_arguments.simulated_arguments.l_global
        self.style_preservation = self.default_arguments.simulated_arguments.l_style_preservation
        self.l_local = self.default_arguments.simulated_arguments.l_local
        self.l_content = self.default_arguments.simulated_arguments.l_content

        # self.style_encoder_adv = self.default_arguments.simulated_arguments.style_encoder_adv
        self.discr_success_th = self.default_arguments.simulated_arguments.discriminator_success_threashold
        self.normal_training_epochs = self.default_arguments.simulated_arguments.normal_training_epochs

        self.global_discr_acc_train = tf.keras.metrics.BinaryAccuracy()
        self.global_discr_acc_valid = tf.keras.metrics.BinaryAccuracy()

        self.local_discr_acc_train = tf.keras.metrics.BinaryAccuracy()
        self.local_discr_acc_valid = tf.keras.metrics.BinaryAccuracy()

        self.content_encoder = ContentEncoder.make_content_encoder(sequence_length, n_signals, feat_wiener)
        self.style_encoder = StyleEncoder.make_style_encoder(sequence_length, n_signals, style_vector_size)
        self.decoder = Decoder.make_generator(n_sample_wiener, feat_wiener, style_vector_size ,n_signals)
        self.global_discriminator = GlobalDiscriminator.make_global_discriminator(sequence_length, n_signals, n_styles)
        self.local_discriminator = LocalDiscriminator.create_local_discriminator(n_signals, sequence_length, n_styles)

        tf.keras.utils.plot_model(self.decoder, show_shapes=True, to_file='Decoder.png')
        tf.keras.utils.plot_model(self.global_discriminator, show_shapes=True, to_file='global_discriminator.png')
        tf.keras.utils.plot_model(self.local_discriminator, show_shapes=True, to_file='local_discriminator.png')
        tf.keras.utils.plot_model(self.content_encoder, show_shapes=True, to_file='content_encoder.png')
        tf.keras.utils.plot_model(self.style_encoder, show_shapes=True, to_file='style_encoder.png')

        # exit()


    def generate(self, content_batch, style_batch):
        content = self.content_encoder(content_batch, training=False)
        style = self.style_encoder(style_batch, training=False)
        generated = self.decoder([content, style], training=False)
        generated = tf.concat(generated, -1)
        return generated
    
    def train(self):
        total_batch_train = "?"
        total_batch_valid = "?"

        discr_success = self.discr_success_th
        discr_success_valid = self.discr_success_th
        alpha = 0.01

        for e in range(self.default_arguments.simulated_arguments.epochs):
            self.logger.reset_metric_states()
            self.logger.reset_valid_states()

            self.global_discr_acc_train.reset_states()
            self.global_discr_acc_valid.reset_states()

            self.local_discr_acc_train.reset_states()
            self.local_discr_acc_valid.reset_states()
            
            
            print("[+] Train Step...")
            for i, (content_batch, style1_sequences, style2_sequences) in enumerate(zip(self.dset_content_train, self.dset_style1_train, self.dset_style2_train)):
                content_sequence1 = content_batch[:int(self.batch_size)]
                content_sequence2 = content_batch[int(self.batch_size):]
                # extra_info = 'train both'
                #e < self.normal_training_epochs

                if e < 0:
                    # let the model train normaly,
                    if i%self.discr_step == 0:
                        global_d_accs, local_d_accs = self.discriminator_step(content_sequence1, style1_sequences, style2_sequences)

                    self.generator_step(content_sequence1, content_sequence2, style1_sequences, style2_sequences)

                    # discr_accs =  central_channel_d_treadoff* global_d_accs + (1- central_channel_d_treadoff)* local_d_accs
                    discr_accs =  np.max([global_d_accs, local_d_accs])
                    discr_success = discr_success * (1. - alpha) + alpha * (discr_accs)
                  
                    self.logger.print_train(e, self.epochs, i, total_batch_train, f"[{global_d_accs:0.2f}; {local_d_accs:0.2f}]: {discr_accs:0.4f} -> {discr_success:0.4f}")
                else: 
                    # Then train one another given their performances.
                    train_d = bool(discr_success < self.discr_success_th)

                    if train_d:
                        test = 'Train Discriminator'
                    else:
                        test = "Train Generator      "

                    global_d_accs, local_d_accs = self.discriminator_step(content_sequence1, style1_sequences, style2_sequences, train_d or i == 0) # ,backward=

                    self.generator_step(content_sequence1, content_sequence2, style1_sequences, style2_sequences, not train_d or i == 0) # , backward=
                    
                    # discr_accs =  central_channel_d_treadoff* global_d_accs + (1- central_channel_d_treadoff)* local_d_accs
                    discr_accs =  np.max([global_d_accs, local_d_accs])
                    discr_success = discr_success * (1. - alpha) + alpha * (discr_accs)

                    self.logger.print_train(e, self.epochs, i, total_batch_train, f"[{global_d_accs:0.2f}; {local_d_accs:0.2f}]: {discr_accs:0.4f} -> {discr_success:0.4f} {test}")
            
            print()
            print("[+] Validation Step...")
            for vb, (content_batch, style1_sequences, style2_sequences) in enumerate(zip(self.dset_content_valid, self.dset_style1_valid, self.dset_style2_valid)):
                content_sequence1 = content_batch[:int(self.batch_size)]
                content_sequence2 = content_batch[int(self.batch_size):]

                global_d_accs, local_d_accs = self.discriminator_valid(content_sequence1, style1_sequences, style2_sequences)

                # discr_accs =  central_channel_d_treadoff* global_d_accs + (1- central_channel_d_treadoff)* local_d_accs
                discr_accs =  np.max([global_d_accs, local_d_accs])

                discr_success_valid = discr_success_valid * (1. - alpha) + alpha * (discr_accs)

                self.generator_valid(content_sequence1, content_sequence2, style1_sequences, style2_sequences)

                self.logger.print_valid(e, self.epochs, vb, total_batch_valid, f"[{global_d_accs:0.2f}; {local_d_accs:0.2f}]: {discr_accs:0.4f} -> {discr_success_valid:0.4f}")
                


            self.training_evaluation(e)
            self.logger.log_train_value("40 - Discriminator Sucess", discr_success, e)            
            self.logger.log_valid_value("40 - Discriminator Sucess", discr_success_valid, e)            

            if e == 0:
                total_batch_train = i
                total_batch_valid = vb
        
        self.save()
    
    def training_evaluation(self, epoch):
        generation_style1_train = self.generate(self.seed_content_train, self.seed_style1_train)
        generation_style1_valid = self.generate(self.seed_content_valid, self.seed_style1_valid)

        generation_style2_train = self.generate(self.seed_content_train, self.seed_style2_train)
        generation_style2_valid = self.generate(self.seed_content_valid, self.seed_style2_valid)

        self.metric_evaluation(generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid)
        self.simple_noise_metric(generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid)
        self.simple_amplitude_metric(generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid)

        plot_buff = self.make_viz()

        self.logger.log_train(plot_buff, epoch)
        self.logger.log_valid(epoch)

    def metric_evaluation(self, generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid):
        metric_s1_train = metric.compute_metric(generation_style1_train, self.seed_style1_train, self.default_arguments.simulated_arguments)
        metric_s1_valid = metric.compute_metric(generation_style1_valid, self.seed_style1_valid, self.default_arguments.simulated_arguments)

        metric_s2_train = metric.compute_metric(generation_style2_train, self.seed_style2_train, self.default_arguments.simulated_arguments)
        metric_s2_valid = metric.compute_metric(generation_style2_valid, self.seed_style2_valid, self.default_arguments.simulated_arguments)
        
        self.logger.met_corr_style1_train(metric_s1_train)
        self.logger.met_corr_style2_train(metric_s2_train)

        self.logger.met_corr_style1_valid(metric_s1_valid)
        self.logger.met_corr_style2_valid(metric_s2_valid)

    def simple_noise_metric(self, generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid):
        # Compute the simple metric specificaly made for noise domain shift.
        # C'est Degeulasse !
        content_gen_style1_train, generated_s1_noise_train = simple_metric.simple_metric_on_noise(generation_style1_train)
        content_gen_style1_valid, generated_s1_noise_valid = simple_metric.simple_metric_on_noise(generation_style1_valid)
        content_gen_style2_train, generated_s2_noise_train = simple_metric.simple_metric_on_noise(generation_style2_train)
        content_gen_style2_valid, generated_s2_noise_valid = simple_metric.simple_metric_on_noise(generation_style2_valid)

        _, seed_style1_noise_train = simple_metric.simple_metric_on_noise(self.seed_style1_train)
        _, seed_style1_noise_valid = simple_metric.simple_metric_on_noise(self.seed_style1_valid)
        _, seed_style2_noise_train = simple_metric.simple_metric_on_noise(self.seed_style2_train)
        _, seed_style2_noise_valid = simple_metric.simple_metric_on_noise(self.seed_style2_valid)

        seed_content_trends_train, _ = simple_metric.simple_metric_on_noise(self.seed_content_train)
        seed_content_trends_valid, _ = simple_metric.simple_metric_on_noise(self.seed_content_valid)

        self.logger.met_noise_sim_style1_train(np.mean(np.abs(seed_style1_noise_train - generated_s1_noise_train)))
        self.logger.met_noise_sim_style1_valid(np.mean(np.abs(seed_style1_noise_valid - generated_s1_noise_valid)))

        self.logger.met_noise_sim_style2_train(np.mean(np.abs(seed_style2_noise_train - generated_s2_noise_train)))
        self.logger.met_noise_sim_style2_valid(np.mean(np.abs(seed_style2_noise_valid - generated_s2_noise_valid)))

        self.logger.met_content_sim_style_1_train(np.mean(np.abs(seed_content_trends_train - content_gen_style1_train)))
        self.logger.met_content_sim_style_2_train(np.mean(np.abs(seed_content_trends_train - content_gen_style2_train)))
        
        self.logger.met_content_sim_style_1_valid(np.mean(np.abs(seed_content_trends_valid - content_gen_style1_valid)))
        self.logger.met_content_sim_style_2_valid(np.mean(np.abs(seed_content_trends_valid - content_gen_style2_valid)))

    def simple_amplitude_metric(self, generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid):
        style1_ampl_diff_train = simple_metric.simple_amplitude_metric(self.seed_style1_train, generation_style1_train)
        style1_ampl_diff_valid = simple_metric.simple_amplitude_metric(self.seed_style1_valid, generation_style1_valid)
        style2_ampl_diff_train = simple_metric.simple_amplitude_metric(self.seed_style2_train, generation_style2_train)
        style2_ampl_diff_valid = simple_metric.simple_amplitude_metric(self.seed_style2_valid, generation_style2_valid)

        self.logger.met_amplitude_sim_style1_train(style1_ampl_diff_train)
        self.logger.met_amplitude_sim_style2_train(style2_ampl_diff_train)
        self.logger.met_amplitude_sim_style1_valid(style1_ampl_diff_valid)
        self.logger.met_amplitude_sim_style2_valid(style2_ampl_diff_valid)

    def make_viz(self):
        vis_fig = visualization_helpers.plot_generated_sequence(
            self.content_encoder, self.style_encoder, self.decoder,
            self.seed_content_valid, 
            self.seed_style1_valid, 
            self.seed_style2_valid,
            config=self.default_arguments.simulated_arguments,
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
            "dset_style_1":self.shell_arguments.style1_dataset, 
            "dset_style_2":self.shell_arguments.style2_dataset, 
            "dset_content":self.shell_arguments.content_dset
        }

        MLFlow_utils.save_configuration(f"{root}/model_config.json", parameters)
        print("[+] Saved !")

    def prepare(self):
        self.seed_content_train = get_batches(self.dset_content_train, 25)
        self.seed_content_valid = get_batches(self.dset_content_valid, 25)

        self.seed_style1_train = get_batches(self.dset_style1_train, 50)
        self.seed_style1_valid = get_batches(self.dset_style1_valid, 50)

        self.seed_style2_train = get_batches(self.dset_style2_train, 50)
        self.seed_style2_valid = get_batches(self.dset_style2_valid, 50)

        self.logger = TensorboardLog(self.shell_arguments)

        self.opt_content_encoder = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.opt_style_encoder = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.opt_decoder = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.local_discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.global_discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

        # self.opt_content_encoder = tf.keras.optimizers.Adam(learning_rate=0.0005) # , beta_1=0. 
        # self.opt_style_encoder = tf.keras.optimizers.Adam(learning_rate=0.0005) # , beta_1=0. 
        # self.opt_decoder = tf.keras.optimizers.Adam(learning_rate=0.0005) # , beta_1=0. 
        # self.local_discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0005) # , beta_1=0. 
        # self.global_discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0005) # , beta_1=0. 

    def instanciate_datasets(self, 
                             content_dset_train:tf.data.Dataset, 
                             content_dset_valid:tf.data.Dataset, 
                             style1_dset_train:tf.data.Dataset, 
                             style1_dset_valid:tf.data.Dataset, 
                             style2_dset_train:tf.data.Dataset,
                             style2_dset_valid:tf.data.Dataset):
        
        self.dset_content_train = content_dset_train
        self.dset_content_valid = content_dset_valid

        self.dset_style1_train = style1_dset_train
        self.dset_style1_valid = style1_dset_valid

        self.dset_style2_train = style2_dset_train
        self.dset_style2_valid = style2_dset_valid

        self.prepare()

    @tf.function
    def discriminator_step(self, content_sequence1, style1_sequences, style2_sequences, backward):
        # Discriminator Step

        with tf.GradientTape(persistent=True) as discr_tape:
            # Sequence generations.
            c1 = self.content_encoder(content_sequence1, training=False)

            s1 = self.style_encoder(style1_sequences, training=False)
            s2 = self.style_encoder(style2_sequences, training=False)

            style1_generated= self.decoder([c1, s1], training=False)
            style2_generated= self.decoder([c1, s2], training=False)

            # split sequences for discriminator's inputs
            style1_sequences_splitted = tf.split(style1_sequences, style1_sequences.shape[-1], axis=-1)
            style2_sequences_splitted = tf.split(style2_sequences, style2_sequences.shape[-1], axis=-1)

            # Global on Real
            g_crit_real1, g_style_classif_real1 = self.global_discriminator(style1_sequences_splitted, training=True)
            g_crit_real2, g_style_classif_real2 = self.global_discriminator(style2_sequences_splitted, training=True)

            # Local on Real
            l_crit1_real = self.local_discriminator(style1_sequences_splitted, training=True)
            l_crit2_real = self.local_discriminator(style2_sequences_splitted, training=True)

            # Global on Generated
            g_crit_fake1, _ = self.global_discriminator(style1_generated, training=True)
            g_crit_fake2, _ = self.global_discriminator(style2_generated, training=True)

            # Local on fake
            l_crit_fake1 = self.local_discriminator(style1_generated, training=True)
            l_crit_fake2 = self.local_discriminator(style2_generated, training=True)

            # Compute the loss for GLOBAL the Discriminator
            g_crit_loss1 = losses.discriminator_loss(g_crit_real1, g_crit_fake1)
            g_crit_loss2 = losses.discriminator_loss(g_crit_real2, g_crit_fake2)

            g_crit_loss = tf.stack((g_crit_loss1, g_crit_loss2))
            g_crit_loss = tf.reduce_mean(g_crit_loss, 0)
            
            style_labels = tf.zeros((self.batch_size, 1))
            g_style1_real = losses.style_classsification_loss(g_style_classif_real1, style_labels+ 0.)
            g_style2_real = losses.style_classsification_loss(g_style_classif_real2, style_labels +1.)
            g_style_real = tf.stack((g_style1_real, g_style2_real))
            g_style_real = tf.reduce_mean(g_style_real, 0)

            l_loss1 = losses.local_discriminator_loss(l_crit1_real, l_crit_fake1)
            l_loss2 = losses.local_discriminator_loss(l_crit2_real, l_crit_fake2)

            l_loss = tf.stack((l_loss1, l_loss2))
            l_loss = tf.reduce_mean(l_loss, 0)

        # (GOBAL DISCRIMINATOR): Real / Fake and style
        global_discr_gradient = discr_tape.gradient([g_crit_loss, g_style_real], self.global_discriminator.trainable_variables)
        grads = discr_tape.gradient(l_loss, self.local_discriminator.trainable_variables)

        if backward:
            self.global_discriminator_opt.apply_gradients(zip(global_discr_gradient, self.global_discriminator.trainable_variables)) 
            self.local_discriminator_opt.apply_gradients(zip(grads, self.local_discriminator.trainable_variables))
    
        self.logger.met_central_d_train(g_crit_loss)
        self.logger.met_central_d_style_real_train(g_style_real)
        self.logger.met_channel_d_train(l_loss)

        # Calculate the performances of the Discriminator.
        real_labels = tf.ones_like(g_crit_real1)
        generation_labels = tf.zeros_like(g_crit_fake1)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.

        global_accs = tf.reduce_mean([
            tf.keras.metrics.binary_accuracy(real_labels, g_crit_real1),
            tf.keras.metrics.binary_accuracy(real_labels, g_crit_real2),
            tf.keras.metrics.binary_accuracy(generation_labels, g_crit_fake1),
            tf.keras.metrics.binary_accuracy(generation_labels, g_crit_fake2)
        ])

        channel_accs = tf.reduce_mean([
            losses.local_discriminator_accuracy(real_labels, l_crit1_real),
            losses.local_discriminator_accuracy(real_labels, l_crit2_real),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake1),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake2)
        ])

        self.logger.met_central_d_accs_train(global_accs)
        self.logger.met_channel_d_accs_train(channel_accs)

        return global_accs, channel_accs


    @tf.function
    def generator_step(self, content_sequence1, content_sequence2, style1_sequences, style2_sequences, backward=True):

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
            contents = tf.concat([content_sequence1, content_sequence2, content_sequence1, content_sequence2], 0)
            styles = tf.concat([style1_sequences, style1_sequences, style2_sequences, style2_sequences], 0)
            _bs = content_sequence1.shape[0]

            encoded_content= self.content_encoder(contents, training=True)
            encoded_styles = self.style_encoder(styles, training=True)

            generations = self.decoder([encoded_content, encoded_styles], training=True)

            merged_generations = tf.concat(generations, -1)

            s_generations = self.style_encoder(merged_generations, training=True)
            c_generations = self.content_encoder(merged_generations, training=True)
            
            style_labels = np.zeros((2* _bs,))
            style_labels[_bs:]= 1.

            # Discriminator pass for the adversarial loss for the generator.
            c1_s1_generations = merged_generations[:_bs]
            c1_s2_generations = merged_generations[2*_bs:3* _bs]

            c1_generations = tf.concat([
                c1_s1_generations,
                c1_s2_generations
            ], 0)

            c1_generations = tf.split(c1_generations, c1_generations.shape[-1], -1)

            crit_on_fake, style_classif_fakes = self.global_discriminator(c1_generations, training=False)

            # Local Discriminator on Fake Data.
            l_crit_on_fake = self.local_discriminator(c1_generations, training=False)

            # Channel Discriminator losses
            local_realness_loss = losses.local_generator_loss(l_crit_on_fake)
            
            # Global Generator losses.
            global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_labels)
            global_realness_loss = losses.generator_loss(crit_on_fake)


            c1s = tf.concat([
                encoded_content[:_bs],                  # 2
                encoded_content[2*_bs:3* _bs]           # 3
            ], 0)

            c2s = tf.concat([
                encoded_content[_bs:2* _bs],            # 2
                encoded_content[3*_bs:]                 # 4 
            ], 0)

            generated_c1s = tf.concat([
                c_generations[:_bs],
                c_generations[2*_bs:3* _bs]
            ], 0)

            generated_c2s = tf.concat([
                c_generations[_bs:2* _bs],
                c_generations[3*_bs:]
            ], 0)

            s_c1_s1 = s_generations[:_bs]
            s_c1_s2 = s_generations[2*_bs: 3*_bs]
            s_c2_s1 = s_generations[_bs: 2*_bs]
            s_c2_s2 = s_generations[3*_bs:] 

            s1s = encoded_styles[:_bs]
            s2s = encoded_styles[2* _bs:3* _bs]

            content_preservation1 = losses.fixed_point_content(c1s, generated_c1s)
            content_preservation2 = losses.fixed_point_content(c2s, generated_c2s)
            content_preservation = (content_preservation1+ content_preservation2)/2

            triplet_style1 =  losses.get_triplet_loss(s1s, s_c1_s1, s_c1_s2, self.default_arguments.simulated_arguments.triplet_r)
            triplet_style2 =  losses.get_triplet_loss(s2s, s_c1_s2, s_c1_s1, self.default_arguments.simulated_arguments.triplet_r)
            triplet_style = (triplet_style1+ triplet_style2)/2

            content_style_disentenglement1 = losses.fixed_point_disentanglement(s_c2_s2, s_c1_s2, s2s)
            content_style_disentenglement2 = losses.fixed_point_disentanglement(s_c2_s1, s_c1_s1, s1s)
            content_style_disentenglement = (content_style_disentenglement1 + content_style_disentenglement2)/2

            content_encoder_loss = self.l_content* content_preservation
            style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement

            g_loss = self.l_reconstr* reconstr_loss+ self.l_global* global_realness_loss + self.style_preservation* global_style_loss+ self.l_local* local_realness_loss

        # Make the Networks Learn!
        content_grad=content_tape.gradient(content_encoder_loss, self.content_encoder.trainable_variables)
        style_grad = style_tape.gradient(style_encoder_loss, self.style_encoder.trainable_variables)
        decoder_grad = decoder_tape.gradient(g_loss, self.decoder.trainable_variables)
            
        if backward == True:
            self.opt_decoder.apply_gradients(zip(decoder_grad, self.decoder.trainable_variables))

        self.opt_content_encoder.apply_gradients(zip(content_grad, self.content_encoder.trainable_variables))
        self.opt_style_encoder.apply_gradients(zip(style_grad, self.style_encoder.trainable_variables))

        self.logger.met_generator_train(g_loss)
        self.logger.met_generator_reconstruction_train(reconstr_loss)

        self.logger.met_generator_local_realness_train(local_realness_loss)
        self.logger.met_generator_global_realness_train(global_realness_loss)

        self.logger.met_central_d_style_fake_train(global_style_loss)

        self.logger.met_disentanglement_train(content_style_disentenglement)
        self.logger.met_triplet_train(triplet_style)
        self.logger.met_style_encoder_train(style_encoder_loss)
        self.logger.met_content_encoder_train(content_preservation)

    @tf.function
    def generator_valid(self, content_sequence1, content_sequence2, style1_sequences, style2_sequences):
        # Reconstruction Loss: Try to generate the same sequence given
        # it's content and style.
        contents = tf.concat([content_sequence1, content_sequence2], 0)
        cs = self.content_encoder(contents, training=True)
        s_cs = self.style_encoder(contents, training=True)
        id_generated = self.decoder([cs, s_cs], training=True)
        id_generated = tf.concat(id_generated, -1)

        reconstr_loss = losses.recontruction_loss(contents, id_generated)

        ####
        contents = tf.concat([content_sequence1, content_sequence2, content_sequence1, content_sequence2], 0)
        styles = tf.concat([style1_sequences, style1_sequences, style2_sequences, style2_sequences], 0)
        _bs = content_sequence1.shape[0]

        encoded_content= self.content_encoder(contents, training=True)
        encoded_styles = self.style_encoder(styles, training=True)

        generations = self.decoder([encoded_content, encoded_styles], training=True)

        merged_generations = tf.concat(generations, -1)

        s_generations = self.style_encoder(merged_generations, training=True)
        c_generations = self.content_encoder(merged_generations, training=True)
            
        style_labels = np.zeros((2* _bs,))
        style_labels[_bs:]= 1.

        # Discriminator pass for the adversarial loss for the generator.
        c1_s1_generations = merged_generations[:_bs]
        c1_s2_generations = merged_generations[2*_bs:3* _bs]

        c1_generations = tf.concat([
            c1_s1_generations,
            c1_s2_generations
        ], 0)

        c1_generations = tf.split(c1_generations, c1_generations.shape[-1], -1)

        crit_on_fake, style_classif_fakes = self.global_discriminator(c1_generations, training=False)

        # Local Discriminator on Fake Data.
        l_crit_on_fake = self.local_discriminator(c1_generations, training=False)

        # Channel Discriminator losses
        local_realness_loss = losses.local_generator_loss(l_crit_on_fake)
        
        # Global Generator losses.
        global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_labels)
        global_realness_loss = losses.generator_loss(crit_on_fake)


        c1s = tf.concat([
            encoded_content[:_bs],                  # 2
            encoded_content[2*_bs:3* _bs]           # 3
        ], 0)

        c2s = tf.concat([
            encoded_content[_bs:2* _bs],            # 2
            encoded_content[3*_bs:]                 # 4 
        ], 0)

        generated_c1s = tf.concat([
            c_generations[:_bs],
            c_generations[2*_bs:3* _bs]
        ], 0)

        generated_c2s = tf.concat([
            c_generations[_bs:2* _bs],
            c_generations[3*_bs:]
        ], 0)

        s_c1_s1 = s_generations[:_bs]
        s_c1_s2 = s_generations[2*_bs: 3*_bs]
        s_c2_s2 = s_generations[3*_bs:] 

        s1s = encoded_styles[:_bs]
        s2s = encoded_styles[2* _bs:3* _bs]

        content_preservation1 = losses.fixed_point_content(c1s, generated_c1s)
        content_preservation2 = losses.fixed_point_content(c2s, generated_c2s)
        content_preservation = (content_preservation1+ content_preservation2)/2

        triplet_style =  losses.get_triplet_loss(s1s, s_c1_s1, s_c1_s2, self.default_arguments.simulated_arguments.triplet_r)
        content_style_disentenglement = losses.fixed_point_disentanglement(s_c2_s2, s_c1_s2, s2s)

        content_encoder_loss = self.l_content* content_preservation
        style_encoder_loss = self.l_triplet* triplet_style + self.l_disentanglement* content_style_disentenglement

        g_loss = self.l_reconstr* reconstr_loss+ self.l_global* global_realness_loss + self.style_preservation* global_style_loss+ self.l_local* local_realness_loss

        self.logger.met_generator_valid(g_loss)
        self.logger.met_generator_reconstruction_valid(reconstr_loss)

        self.logger.met_generator_local_realness_valid(local_realness_loss)
        self.logger.met_generator_global_realness_valid(global_realness_loss)

        self.logger.met_central_d_style_fake_valid(global_style_loss)

        self.logger.met_content_encoder_valid(content_preservation)
        
        self.logger.met_style_encoder_valid(style_encoder_loss)
        self.logger.met_triplet_valid(triplet_style)
        self.logger.met_disentanglement_valid(content_style_disentenglement)



    @tf.function
    def discriminator_valid(self, content_sequence1, style1_sequences, style2_sequences):
        c1 = self.content_encoder(content_sequence1, training=False)

        s1 = self.style_encoder(style1_sequences, training=False)
        s2 = self.style_encoder(style2_sequences, training=False)

        style1_generated= self.decoder([c1, s1], training=False)
        style2_generated= self.decoder([c1, s2], training=False)


        # split sequences for discriminator's inputs
        style1_sequences_splitted = tf.split(style1_sequences, style1_sequences.shape[-1], axis=-1)
        style2_sequences_splitted = tf.split(style2_sequences, style2_sequences.shape[-1], axis=-1)

        # Global on Real
        g_crit_real1, g_style_classif_real1 = self.global_discriminator(style1_sequences_splitted, training=True)
        g_crit_real2, g_style_classif_real2 = self.global_discriminator(style2_sequences_splitted, training=True)

        # Local on Real
        l_crit1_real = self.local_discriminator(style1_sequences_splitted, training=True)
        l_crit2_real = self.local_discriminator(style2_sequences_splitted, training=True)

        # Global on Generated
        g_crit_fake1, _ = self.global_discriminator(style1_generated, training=True)
        g_crit_fake2, _ = self.global_discriminator(style2_generated, training=True)

        # Local on fake
        l_crit_fake1 = self.local_discriminator(style1_generated, training=True)
        l_crit_fake2 = self.local_discriminator(style2_generated, training=True)

        # Compute the loss for GLOBAL the Discriminator
        g_crit_loss1 = losses.discriminator_loss(g_crit_real1, g_crit_fake1)
        g_crit_loss2 = losses.discriminator_loss(g_crit_real2, g_crit_fake2)

        g_crit_loss = tf.stack((g_crit_loss1, g_crit_loss2))
        g_crit_loss = tf.reduce_mean(g_crit_loss, 0)
        
        style_labels = tf.zeros((self.batch_size, 1))
        g_style1_real = losses.style_classsification_loss(g_style_classif_real1, style_labels+ 0.)
        g_style2_real = losses.style_classsification_loss(g_style_classif_real2, style_labels +1.)
        g_style_real = tf.stack((g_style1_real, g_style2_real))
        g_style_real = tf.reduce_mean(g_style_real, 0)

        l_loss1 = losses.local_discriminator_loss(l_crit1_real, l_crit_fake1)
        l_loss2 = losses.local_discriminator_loss(l_crit2_real, l_crit_fake2)

        l_loss = tf.stack((l_loss1, l_loss2))
        l_loss = tf.reduce_mean(l_loss, 0)

        self.logger.met_central_d_valid(g_crit_loss)
        self.logger.met_central_d_style_real_valid(g_style_real)

        self.logger.met_channel_d_valid(l_loss)

        # Calculate the performances of the Discriminator.
        real_labels = tf.ones_like(g_crit_fake1)
        generation_labels = tf.zeros_like(g_crit_fake1)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.
        global_accs = tf.reduce_mean([
            tf.keras.metrics.binary_accuracy(real_labels, g_crit_real1),
            tf.keras.metrics.binary_accuracy(real_labels, g_crit_real2),
            tf.keras.metrics.binary_accuracy(generation_labels, g_crit_fake1),
            tf.keras.metrics.binary_accuracy(generation_labels, g_crit_fake2)
        ])

        channel_accs = tf.reduce_mean([
            losses.local_discriminator_accuracy(real_labels, l_crit1_real),
            losses.local_discriminator_accuracy(real_labels, l_crit2_real),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake1),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake2)
        ])

        self.logger.met_central_d_accs_valid(global_accs)
        self.logger.met_channel_d_accs_valid(channel_accs)

        return global_accs, channel_accs
    
