import matplotlib.pyplot as plt
import numpy as np
import io

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

from datetime import datetime



class TensorboardLog():
    def __init__(self, shell_arguments) -> None:
        root = shell_arguments.tensorboard_root
    
        date_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        exp_name = shell_arguments.exp_name

        self.full_path = f"{root}/{date_str} - {exp_name}"

        self.train_summary_writer = tf.summary.create_file_writer(self.full_path + "/train")
        self.valid_summary_writer = tf.summary.create_file_writer(self.full_path + "/valid")

        self.intanciate_loggers()


    def log_train(self, plot_buf, epoch):
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self.train_summary_writer.as_default():
            tf.summary.scalar("00 - Correlation Metric Style 1", self.met_corr_style1_train.result(), step=epoch)
            tf.summary.scalar("01 - Correlation Metric Style 2", self.met_corr_style2_train.result(), step=epoch)

            tf.summary.scalar("02 - Noise Similarity Style 1", self.met_noise_sim_style1_train.result(), step=epoch)
            tf.summary.scalar("03 - Noise Similarity Style 2", self.met_noise_sim_style2_train.result(), step=epoch)

            tf.summary.scalar("04 - Content Similarity Style 1", self.met_content_sim_style_1_train.result(), step=epoch)
            tf.summary.scalar("05 - Content Similarity Style 2", self.met_content_sim_style_2_train.result(), step=epoch)
            
            tf.summary.scalar("06 - Amplitude Similarity Style 1", self.met_amplitude_sim_style1_train.result(), step=epoch)
            tf.summary.scalar("07 - Amplitude Similarity Style 2", self.met_amplitude_sim_style2_train.result(), step=epoch)

            tf.summary.scalar("10 - Total Generator Loss", self.met_generator_train.result(), step=epoch)
            tf.summary.scalar("11 - Reconstruction from Content", self.met_generator_reconstruction_train.result(), step=epoch)

            tf.summary.scalar("12 - Central Realness", self.met_generator_global_realness_train.result(), step=epoch)
            tf.summary.scalar("13 - Local Realness", self.met_generator_local_realness_train.result(), step=epoch)

            tf.summary.scalar("20 - Style Loss", self.met_style_encoder_train.result(), step=epoch)
            tf.summary.scalar("21 - Triplet Loss", self.met_triplet_train.result(), step=epoch)
            tf.summary.scalar("22 - Disentanglement Loss", self.met_disentanglement_train.result(), step=epoch)

            tf.summary.scalar("30 - Content Loss", self.met_content_encoder_train.result(), step=epoch)

            tf.summary.scalar("40 - Global Discriminator Loss", self.met_central_d_train.result(), step=epoch)
            tf.summary.scalar("40 - Global Discriminator Acc", self.met_central_d_accs_train.result(), step=epoch)

            tf.summary.scalar("40 - Local Discriminator Loss", self.met_channel_d_train.result(), step=epoch)
            tf.summary.scalar("40 - Local Discriminator Acc", self.met_channel_d_accs_train.result(), step=epoch)

            tf.summary.scalar("41 - Global Discriminator Style Loss (Real Data)", self.met_central_d_style_real_train.result(), step=epoch)
            tf.summary.scalar("41 - Global Discriminator Style Loss (Fake Data)", self.met_central_d_style_fake_train.result(), step=epoch)

            tf.summary.scalar("42 - Local Discriminator Style Loss (Real Data)", self.met_channel_d_style_real_train.result(), step=epoch)
            tf.summary.scalar("42 - Local Discriminator Style Loss (Fake Data)", self.met_channel_d_style_fake_train.result(), step=epoch)



            tf.summary.image("Training Generations", image, step=epoch)

    def log_valid(self, epoch):

        with self.valid_summary_writer.as_default():
            tf.summary.scalar("00 - Correlation Metric Style 1", self.met_corr_style1_valid.result(), step=epoch)
            tf.summary.scalar("01 - Correlation Metric Style 2", self.met_corr_style2_valid.result(), step=epoch)

            tf.summary.scalar("02 - Noise Similarity Style 1", self.met_noise_sim_style1_valid.result(), step=epoch)
            tf.summary.scalar("03 - Noise Similarity Style 2", self.met_noise_sim_style2_valid.result(), step=epoch)

            tf.summary.scalar("04 - Content Similarity Style 1", self.met_content_sim_style_1_valid.result(), step=epoch)
            tf.summary.scalar("05 - Content Similarity Style 2", self.met_content_sim_style_2_valid.result(), step=epoch)

            tf.summary.scalar("06 - Amplitude Similarity Style 1", self.met_amplitude_sim_style1_valid.result(), step=epoch)
            tf.summary.scalar("07 - Amplitude Similarity Style 2", self.met_amplitude_sim_style2_valid.result(), step=epoch)

            tf.summary.scalar("10 - Total Generator Loss", self.met_generator_valid.result(), step=epoch)
            tf.summary.scalar("11 - Reconstruction from Content", self.met_generator_reconstruction_valid.result(), step=epoch)

            tf.summary.scalar("12 - Central Realness", self.met_generator_global_realness_valid.result(), step=epoch)
            tf.summary.scalar("13 - Local Realness", self.met_generator_local_realness_valid.result(), step=epoch)

            tf.summary.scalar("20 - Style Loss", self.met_style_encoder_valid.result(), step=epoch)
            tf.summary.scalar("21 - Triplet Loss", self.met_triplet_valid.result(), step=epoch)
            tf.summary.scalar("22 - Disentanglement Loss", self.met_disentanglement_valid.result(), step=epoch)

            tf.summary.scalar("30 - Content Loss", self.met_content_encoder_valid.result(), step=epoch)

            tf.summary.scalar("40 - Global Discriminator Loss", self.met_central_d_valid.result(), step=epoch)
            tf.summary.scalar("40 - Global Discriminator Acc", self.met_central_d_accs_valid.result(), step=epoch)


            tf.summary.scalar("40 - Local Discriminator Loss", self.met_channel_d_valid.result(), step=epoch)
            tf.summary.scalar("40 - Local Discriminator Acc", self.met_channel_d_accs_valid.result(), step=epoch)

            tf.summary.scalar("41 - Global Discriminator Style Loss (Real Data)", self.met_central_d_style_real_valid.result(), step=epoch)
            tf.summary.scalar("41 - Global Discriminator Style Loss (Fake Data)", self.met_central_d_style_fake_valid.result(), step=epoch)

            tf.summary.scalar("42 - Local Discriminator Style Loss (Real Data)", self.met_channel_d_style_real_valid.result(), step=epoch)
            tf.summary.scalar("42 - Local Discriminator Style Loss (Fake Data)", self.met_channel_d_style_fake_valid.result(), step=epoch)




    def log_train_value(self, key, value, step):
        with self.train_summary_writer.as_default():
            tf.summary.scalar(key, value, step=step)

    def log_valid_value(self, key, value, step):
        with self.valid_summary_writer.as_default():
            tf.summary.scalar(key, value, step=step)

    def reset_metric_states(self):
        self.met_generator_train.reset_states()
        self.met_generator_reconstruction_train.reset_states()
        self.met_generator_local_realness_train.reset_states()
        self.met_generator_global_realness_train.reset_states()
        self.met_triplet_train.reset_states()
        self.met_disentanglement_train.reset_states()
        self.met_style_encoder_train.reset_states()
        self.met_content_encoder_train.reset_states()
        self.met_central_d_train.reset_states()
        self.met_central_d_style_real_train.reset_states()
        self.met_central_d_style_fake_train.reset_states()
        self.met_channel_d_train.reset_states()
        self.met_channel_d_style_real_train.reset_states()
        self.met_channel_d_style_fake_train.reset_states()
        self.met_corr_style1_train.reset_states()
        self.met_corr_style2_train.reset_states()
        self.met_noise_sim_style1_train.reset_states()
        self.met_noise_sim_style2_train.reset_states()
    
        self.met_amplitude_sim_style1_train.reset_states()
        self.met_amplitude_sim_style2_train.reset_states()

        self.met_central_d_accs_train.reset_states()
        self.met_channel_d_accs_train.reset_states()

    def reset_valid_states(self):
        self.met_generator_valid.reset_states()
        self.met_generator_reconstruction_valid.reset_states()
        self.met_generator_local_realness_valid.reset_states()
        self.met_generator_global_realness_valid.reset_states()
        self.met_triplet_valid.reset_states()
        self.met_disentanglement_valid.reset_states()
        self.met_style_encoder_valid.reset_states()
        self.met_content_encoder_valid.reset_states()
        self.met_central_d_valid.reset_states()
        self.met_central_d_style_real_valid.reset_states()
        self.met_central_d_style_fake_valid.reset_states()
        self.met_channel_d_valid.reset_states()
        self.met_channel_d_style_real_valid.reset_states()
        self.met_channel_d_style_fake_valid.reset_states()
        self.met_corr_style1_valid.reset_states()
        self.met_corr_style2_valid.reset_states()
        self.met_noise_sim_style1_valid.reset_states()
        self.met_noise_sim_style2_valid.reset_states()
        self.met_amplitude_sim_style1_valid.reset_states()
        self.met_amplitude_sim_style2_valid.reset_states()

        self.met_central_d_accs_valid.reset_states()
        self.met_channel_d_accs_valid.reset_states()
        

    def intanciate_loggers(self):
        self.met_generator_train = tf.keras.metrics.Mean(name="Total Generator Loss")

        self.met_generator_reconstruction_train = tf.keras.metrics.Mean(name="Generator Reconstruction Loss")

        self.met_generator_local_realness_train = tf.keras.metrics.Mean(name="Generator local Realness loss Train")
        self.met_generator_global_realness_train = tf.keras.metrics.Mean(name="Generator Global Realness loss Train")

        # Style Encoder Loss
        self.met_triplet_train = tf.keras.metrics.Mean(name="Total Triplet Loss")
        self.met_disentanglement_train = tf.keras.metrics.Mean(name="Disentanglement Loss")
        self.met_style_encoder_train = tf.keras.metrics.Mean(name="Style Loss")

        # Content encoder Loss
        self.met_content_encoder_train= tf.keras.metrics.Mean(name="Content Encoder Loss")

        # Central Discriminator
        self.met_central_d_train= tf.keras.metrics.Mean(name="Central Discriminator Loss")
        self.met_central_d_style_real_train = tf.keras.metrics.Mean(name="Central Discriminator Loss Real Style Classif")
        self.met_central_d_style_fake_train = tf.keras.metrics.Mean(name="Central Discriminator Loss Fake Style Classif")

        # Channel Discriminator.
        self.met_channel_d_train= tf.keras.metrics.Mean(name="Channel Discriminator Loss")
        self.met_channel_d_style_real_train = tf.keras.metrics.Mean(name="Channel Discriminator Real Style Classification")
        self.met_channel_d_style_fake_train = tf.keras.metrics.Mean(name="Channel Discriminator Fake Style Classification")

        # Correlation Metric
        self.met_corr_style1_train = tf.keras.metrics.Mean(name="Correlation Metric Style 1")
        self.met_corr_style2_train = tf.keras.metrics.Mean(name="Correlation Metric Style 2")

        # Noise Extraction Metric
        self.met_noise_sim_style1_train = tf.keras.metrics.Mean(name="Noise Similarity Style1.")
        self.met_noise_sim_style2_train = tf.keras.metrics.Mean(name="Noise Similarity Style2.")

        self.met_content_sim_style_1_train = tf.keras.metrics.Mean(name="Content Similarity Style1.")
        self.met_content_sim_style_2_train = tf.keras.metrics.Mean(name="Content Similarity Style2.")

        self.met_amplitude_sim_style1_train= tf.keras.metrics.Mean(name="Amplitude Similarity Style1.")
        self.met_amplitude_sim_style2_train= tf.keras.metrics.Mean(name="Amplitude Similarity Style2.")

        # Valid Metrics
        # Generator Metric
        self.met_generator_valid = tf.keras.metrics.Mean(name="Total Generator Loss")

        self.met_generator_reconstruction_valid = tf.keras.metrics.Mean(name="Generator Reconstruction Loss")

        self.met_generator_local_realness_valid = tf.keras.metrics.Mean(name="Generator local Realness loss valid")
        self.met_generator_global_realness_valid = tf.keras.metrics.Mean(name="Generator Global Realness loss valid")

        # Style Encoder Loss
        self.met_triplet_valid = tf.keras.metrics.Mean(name="Total Triplet Loss")
        self.met_disentanglement_valid = tf.keras.metrics.Mean(name="Disentanglement Loss")
        self.met_style_encoder_valid = tf.keras.metrics.Mean(name="Style Loss")

        # Content encoder Loss
        self.met_content_encoder_valid= tf.keras.metrics.Mean(name="Content Encoder Loss")

        # Central Discriminator
        self.met_central_d_valid= tf.keras.metrics.Mean(name="Central Discriminator Loss")
        self.met_central_d_style_real_valid = tf.keras.metrics.Mean(name="Central Discriminator Loss Real Style Classif")
        self.met_central_d_style_fake_valid = tf.keras.metrics.Mean(name="Central Discriminator Loss Fake Style Classif")

        # Channel Discriminator.
        self.met_channel_d_valid= tf.keras.metrics.Mean(name="Channel Discriminator Loss")
        self.met_channel_d_style_real_valid = tf.keras.metrics.Mean(name="Channel Discriminator Real Style Classification")
        self.met_channel_d_style_fake_valid = tf.keras.metrics.Mean(name="Channel Discriminator Fake Style Classification")

        # Correlation Metric
        self.met_corr_style1_valid = tf.keras.metrics.Mean(name="Correlation Metric Style 1")
        self.met_corr_style2_valid = tf.keras.metrics.Mean(name="Correlation Metric Style 2")

        self.met_noise_sim_style1_valid = tf.keras.metrics.Mean(name="Noise Similarity Style1.")
        self.met_noise_sim_style2_valid = tf.keras.metrics.Mean(name="Noise Similarity Style2.")

        self.met_content_sim_style_1_valid = tf.keras.metrics.Mean(name="Content Similarity Style1.")
        self.met_content_sim_style_2_valid = tf.keras.metrics.Mean(name="Content Similarity Style2.")

        self.met_amplitude_sim_style1_valid= tf.keras.metrics.Mean(name="Amplitude Similarity Style1.")
        self.met_amplitude_sim_style2_valid= tf.keras.metrics.Mean(name="Amplitude Similarity Style2.")

        self.met_channel_d_accs_train = tf.keras.metrics.Mean(name="Channel Discriminator Accuracy.")
        self.met_central_d_accs_train = tf.keras.metrics.Mean(name="Channel Discriminator Accuracy.")

        self.met_channel_d_accs_valid = tf.keras.metrics.Mean(name="Channel Discriminator Accuracy.")
        self.met_central_d_accs_valid = tf.keras.metrics.Mean(name="Channel Discriminator Accuracy.")


    def print_train(self, epoch, total_epochs, i, total_batch, extra_info:str=""):
        # print(f"\r e:{epoch}/{total_epochs}; {i}/{total_batch}. G_loss {self.met_generator_train.result():0.2f}; Triplet Loss {self.met_triplet_train.result():0.2f}; Disentanglement Loss: {self.met_disentanglement_train.result():0.2f}; Content Loss {self.met_content_encoder_train.result():0.2f} Local D [Crit; Style]: [{self.met_channel_d_train.result():0.2f}; {self.met_channel_d_style_real_train.result():0.2f}]; Global D [Crit; Style]: [{self.met_central_d_train.result():0.2f}; {self.met_central_d_style_fake_train.result():0.2f}] {extra_info}       ", end="")
        print(f"\r e:{epoch}/{total_epochs}; {i}/{total_batch}. G_loss {self.met_generator_train.result():0.2f}; Local D [Crit; Style]: [{self.met_channel_d_train.result():0.2f}; {self.met_channel_d_style_real_train.result():0.2f}]; Global D [Crit; Style]: [{self.met_central_d_train.result():0.2f}; {self.met_central_d_style_fake_train.result():0.2f}] {extra_info}       ", end="")

    def print_valid(self, e, epochs, vb, total_batch, extra_info:str=""):
        # print(f"\r e:{e}/{epochs}; {vb}/{total_batch}. G_loss {self.met_generator_valid.result():0.2f}; Triplet Loss {self.met_triplet_valid.result():0.2f}; Disentanglement Loss: {self.met_disentanglement_valid.result():0.2f}; Content Loss {self.met_content_encoder_valid.result():0.2f} Local D [Crit; Style]: [{self.met_channel_d_valid.result():0.2f}; {self.met_channel_d_style_real_valid.result():0.2f}]; Global D [Crit; Style]: [{self.met_central_d_valid.result():0.2f}; {self.met_central_d_style_fake_valid.result():0.2f}] {extra_info}      ", end="")
        print(f"\r e:{e}/{epochs}; {vb}/{total_batch}. G_loss {self.met_generator_valid.result():0.2f}; Local D [Crit; Style]: [{self.met_channel_d_valid.result():0.2f}; {self.met_channel_d_style_real_valid.result():0.2f}]; Global D [Crit; Style]: [{self.met_central_d_valid.result():0.2f}; {self.met_central_d_style_fake_valid.result():0.2f}] {extra_info}      ", end="")

    def fig_to_buff(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

