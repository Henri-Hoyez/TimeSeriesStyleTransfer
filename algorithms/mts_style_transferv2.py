from models.mtsStyleTransferV1 import ContentEncoder, Decoder, GlobalDiscriminator, LocalDiscriminator, StyleEncoder
from utils.dataLoader import get_batches
from utils.tensorboard_log import TensorboardLog
from utils import metric, simple_metric, visualization_helpersv2, MLFlow_utils

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

        # self.style_encoder.summary()


    def plot_models(self):
        tf.keras.utils.plot_model(self.decoder, show_shapes=True, to_file='Decoder.png')
        tf.keras.utils.plot_model(self.global_discriminator, show_shapes=True, to_file='global_discriminator.png')
        tf.keras.utils.plot_model(self.local_discriminator, show_shapes=True, to_file='local_discriminator.png')
        tf.keras.utils.plot_model(self.content_encoder, show_shapes=True, to_file='content_encoder.png')
        tf.keras.utils.plot_model(self.style_encoder, show_shapes=True, to_file='style_encoder.png')


    

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
            for i, (content_batch, style_batch) in enumerate(zip(self.dset_content_train, self.dsets_style_train)):

                content_sequence1 = content_batch[:int(self.batch_size)]
                content_sequence2 = content_batch[int(self.batch_size):]
    
                # Then train one another given their performances.
                train_d = bool(discr_success < self.discr_success_th)

                if train_d:
                    test = 'Train Discriminator'
                else:
                    test = "Train Generator      "

                global_d_accs, local_d_accs = self.discriminator_step(content_sequence1, style_batch, train_d or i == 0)

                self.generator_step(content_sequence1, content_sequence2, style_batch, not train_d or i == 0) # , backward=
                
                # discr_accs =  central_channel_d_treadoff* global_d_accs + (1- central_channel_d_treadoff)* local_d_accs
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
    
    def training_evaluation(self, epoch):
        generation_style1_train = self.generate(self.seed_content_train, self.seed_styles_train[0])
        generation_style1_valid = self.generate(self.seed_content_valid, self.seed_styles_valid[0])

        generation_style2_train = self.generate(self.seed_content_train, self.seed_styles_train[1])
        generation_style2_valid = self.generate(self.seed_content_valid, self.seed_styles_valid[1])

        self.metric_evaluation(generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid)
        self.simple_noise_metric(generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid)
        self.simple_amplitude_metric(generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid)

        plot_buff = self.make_viz()

        self.logger.log_train(plot_buff, epoch)
        self.logger.log_valid(epoch)

    def metric_evaluation(self, generation_style1_train, generation_style1_valid, generation_style2_train, generation_style2_valid):
        metric_s1_train = metric.compute_metric(generation_style1_train, self.seed_styles_train[0], self.default_arguments.simulated_arguments)
        metric_s1_valid = metric.compute_metric(generation_style1_valid, self.seed_styles_valid[0], self.default_arguments.simulated_arguments)

        metric_s2_train = metric.compute_metric(generation_style2_train, self.seed_styles_train[1], self.default_arguments.simulated_arguments)
        metric_s2_valid = metric.compute_metric(generation_style2_valid, self.seed_styles_valid[1], self.default_arguments.simulated_arguments)
        
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

        _, seed_style1_noise_train = simple_metric.simple_metric_on_noise(self.seed_styles_train[0])
        _, seed_style1_noise_valid = simple_metric.simple_metric_on_noise(self.seed_styles_valid[0])
        _, seed_style2_noise_train = simple_metric.simple_metric_on_noise(self.seed_styles_train[1])
        _, seed_style2_noise_valid = simple_metric.simple_metric_on_noise(self.seed_styles_valid[1])

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
        style1_ampl_diff_train = simple_metric.simple_amplitude_metric(self.seed_styles_train[0], generation_style1_train)
        style1_ampl_diff_valid = simple_metric.simple_amplitude_metric(self.seed_styles_valid[0], generation_style1_valid)
        style2_ampl_diff_train = simple_metric.simple_amplitude_metric(self.seed_styles_train[1], generation_style2_train)
        style2_ampl_diff_valid = simple_metric.simple_amplitude_metric(self.seed_styles_valid[1], generation_style2_valid)

        self.logger.met_amplitude_sim_style1_train(style1_ampl_diff_train)
        self.logger.met_amplitude_sim_style2_train(style2_ampl_diff_train)
        self.logger.met_amplitude_sim_style1_valid(style1_ampl_diff_valid)
        self.logger.met_amplitude_sim_style2_valid(style2_ampl_diff_valid)

    def make_viz(self):
        # vis_fig = visualization_helpersv2.plot_generated_sequence(
        #     self.content_encoder, self.style_encoder, self.decoder,
        #     self.seed_content_valid, 
        #     self.seed_styles_valid[0], 
        #     self.seed_styles_valid[1],
        #     config=self.default_arguments.simulated_arguments,
        #     )
        
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
            "dset_style_1":self.shell_arguments.style1_dataset, 
            "dset_style_2":self.shell_arguments.style2_dataset, 
            "dset_content":self.shell_arguments.content_dset
        }

        MLFlow_utils.save_configuration(f"{root}/model_config.json", parameters)
        print("[+] Saved !")

    def set_seeds(self, _seed_style_train, _seed_style_valid):

        self.seed_styles_train = _seed_style_train
        self.seed_styles_valid = _seed_style_valid
        
        self.seed_content_train = get_batches(self.dset_content_train, 25)
        self.seed_content_valid = get_batches(self.dset_content_valid, 25)


    def prepare(self):

        self.logger = TensorboardLog(self.shell_arguments)

        self.opt_content_encoder = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.opt_style_encoder = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.opt_decoder = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.local_discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
        self.global_discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=0.0005) 

    def instanciate_datasets(self, 
                             content_dset_train:tf.data.Dataset, 
                             content_dset_valid:tf.data.Dataset, 
                             styles_dset_train:tf.data.Dataset, 
                             styles_dset_valid:tf.data.Dataset):
        
        self.dset_content_train = content_dset_train
        self.dset_content_valid = content_dset_valid

        self.dsets_style_train = styles_dset_train
        self.dsets_style_valid = styles_dset_valid

        self.prepare()

    @tf.function
    def discriminator_step(self, content_sequence1, style_batch, backward):
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
            g_crit_loss = losses.discriminator_loss(g_crit_real, g_crit_fake)
            g_style_real = losses.style_classsification_loss(g_style_classif_real, style_labels)

            l_loss = losses.local_discriminator_loss(l_crit_real, l_crit_fake)

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
        real_labels = tf.ones_like(g_crit_real)
        generation_labels = tf.zeros_like(g_crit_fake)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.

        global_accs = tf.reduce_mean([
            tf.keras.metrics.binary_accuracy(real_labels, g_crit_real),
            tf.keras.metrics.binary_accuracy(generation_labels, g_crit_fake)
        ])

        channel_accs = tf.reduce_mean([
            losses.local_discriminator_accuracy(real_labels, l_crit_real),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake)
        ])

        self.logger.met_central_d_accs_train(global_accs)
        self.logger.met_channel_d_accs_train(channel_accs)

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
            local_realness_loss = losses.local_generator_loss(l_crit_on_fake)
            
            # Global Generator losses.

            global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_label_extended)
            global_realness_loss = losses.generator_loss(crit_on_fake)

            ########
            content_preservation = losses.fixed_point_content(encoded_content, c_generations)

            s_c1_s = s_generations[:_bs]
            s_c2_s = s_generations[_bs:]

            content_style_disentenglement = losses.fixed_point_disentanglement(s_c2_s, s_c1_s, encoded_styles[:_bs])

            # triplet_style =  losses.get_triplet_loss(anchor, positive, negative, self.default_arguments.simulated_arguments.triplet_r)
            triplet_style1 = losses.hard_triplet(style_labels, s_c1_s)
            triplet_style2 = losses.hard_triplet(style_labels, s_c2_s)
            triplet_style = (triplet_style1+ triplet_style2)/2

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
    def generator_valid(self, content_sequence1, content_sequence2, style_batch):
        # Reconstruction Loss: Try to generate the same sequence given
        # it's content and style.
        style_sequences, style_labels = style_batch[0],  style_batch[1]

        # Here, things get a little bit more complicated :)
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
        local_realness_loss = losses.local_generator_loss(l_crit_on_fake)
        
        # Global Generator losses.

        global_style_loss = losses.style_classsification_loss(style_classif_fakes, style_label_extended)
        global_realness_loss = losses.generator_loss(crit_on_fake)

        ########
        content_preservation = losses.fixed_point_content(encoded_content, c_generations)

        s_c1_s = s_generations[:_bs]
        s_c2_s = s_generations[_bs:]

        content_style_disentenglement = losses.fixed_point_disentanglement(s_c2_s, s_c1_s, encoded_styles[:_bs])

        triplet_style1 = losses.hard_triplet(style_labels, s_c1_s)
        triplet_style2 = losses.hard_triplet(style_labels, s_c2_s)
        triplet_style = (triplet_style1+ triplet_style2)/2


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
        g_crit_loss = losses.discriminator_loss(g_crit_real, g_crit_fake)
        g_style_real = losses.style_classsification_loss(g_style_classif_real, style_labels)

        l_loss = losses.local_discriminator_loss(l_crit_real, l_crit_fake)

        self.logger.met_central_d_valid(g_crit_loss)
        self.logger.met_central_d_style_real_valid(g_style_real)

        self.logger.met_channel_d_valid(l_loss)

        # Calculate the performances of the Discriminator.
        real_labels = tf.ones_like(g_crit_real)
        generation_labels = tf.zeros_like(g_crit_fake)

        # Compute Accuracy for the GAN Training.
        # Thie accuracy will define if the generator has to be trained or the discriminator.
        global_accs = tf.reduce_mean([
            tf.keras.metrics.binary_accuracy(real_labels, g_crit_real),
            tf.keras.metrics.binary_accuracy(generation_labels, g_crit_fake)
        ])

        channel_accs = tf.reduce_mean([
            losses.local_discriminator_accuracy(real_labels, l_crit_real),
            losses.local_discriminator_accuracy(generation_labels, l_crit_fake)
        ])

        self.logger.met_central_d_accs_valid(global_accs)
        self.logger.met_channel_d_accs_valid(channel_accs)

        return global_accs, channel_accs
    
