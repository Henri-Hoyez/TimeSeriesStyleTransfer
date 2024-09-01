import numpy as np
from utils.metric import signature_on_batch
from configs.SimulatedData import Proposed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf


def draw_arrow(A, B, ax:plt.Axes, color="b", width=0.001):
    ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              length_includes_head=True, color=color, width=width, alpha=0.95, head_width=width*4.0)
    
def draw_arrows(xs, ys, ax:plt.Axes, color="b"):
    # points = np.stack((xs, ys)).T
    # dist = points[1:] - points[:-1]

    # norms = np.linalg.norm(dist, axis=1)
    # norm = np.mean(norms)

    for i in range(xs.shape[0]-1):
        point0 = [xs[i], ys[i]]
        point1 = [xs[i+1], ys[i+1]]
        draw_arrow(point0, point1, ax, color=color, width=0.00025)
        
        
def draw_content_space(
    ax:plt.Axes,
    content_wiener_process:tf.Tensor,
    color='tab:blue',
    label='An amayzing label.'):
    
    ax.scatter(content_wiener_process[:, 0], content_wiener_process[:, 1], label=label)
    draw_arrows(content_wiener_process[:, 0], content_wiener_process[:, 1], ax, color)
    
    
    



def plot_generated_sequence(
        content_encoder, style_encoder, decoder, 
        content_sequences, 
        seed_style_sequences:tf.Tensor, title="Default Title."):
    
    # seed style sequence:
    # Shape [n_style, nsequence, nfeat]

    # Generate viz for plots 
    style_index = 0
    n_style = seed_style_sequences.shape[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    c = np.array([content_sequences[0]])
    c = content_encoder(c)
    content_encoded = tf.concat([c]*seed_style_sequences.shape[0], 0)
    
    reshaped_style_sequences = tf.reshape(seed_style_sequences, (-1, seed_style_sequences.shape[-2], seed_style_sequences.shape[-1]))

    style_encoded = style_encoder(reshaped_style_sequences)
    
    pca = PCA(2)
    
    reduced_style = pca.fit_transform(style_encoded)
    
    reduced_style = tf.reshape(reduced_style, (
        seed_style_sequences.shape[0],
        style_encoded.shape[0]//seed_style_sequences.shape[0], 
        -1))
    
    reshaped_style_encoded = tf.reshape(style_encoded, (
        seed_style_sequences.shape[0],
        style_encoded.shape[0]//seed_style_sequences.shape[0], 
        -1))

    generated_viz = decoder([content_encoded, reshaped_style_encoded[:, style_index, :]])
    
    generated_viz = tf.concat(generated_viz, -1)
    
    content_of_generated_viz = content_encoder(generated_viz)
    
    
    # Make point for Style Scatter plot. 
    c1s = np.array([c[0]]* reshaped_style_sequences.shape[0])
    
    print(c1s.shape, style_encoded.shape)
    
    gen_c1_s = decoder([c1s, style_encoded])
    gen_c1_s = tf.concat(gen_c1_s, -1)
    
    style_of_generated_viz = style_encoder(gen_c1_s)
    
    reduced_style_of_generated_viz = pca.transform(style_of_generated_viz)
    print(gen_c1_s.shape, style_of_generated_viz.shape)
        
    reduced_style_of_generated_viz = tf.reshape(reduced_style_of_generated_viz, (
        seed_style_sequences.shape[0],
        style_encoded.shape[0]//seed_style_sequences.shape[0], 
        -1))
    
    
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=25)
    spec= fig.add_gridspec(3, 8)
    
    
    ax00 = fig.add_subplot(spec[0:2, :2])
    ax00.set_title('Content Sequence. ($C_0$)')
    ax00.plot(content_sequences[0])
    ax00.grid(True)
    ax00.legend()

# #######
    ax01 = fig.add_subplot(spec[0, 2:5])
    ax01.set_title('Style Sequence 1. ($S_0$)')
    ax01.plot(seed_style_sequences[0, style_index, :])
    ax01.grid(True)

    ax11 = fig.add_subplot(spec[1, 2:5])
    ax11.set_title("Style Sequence 2. ($S_1$)")
    ax11.plot(seed_style_sequences[1, style_index, :])
    ax11.grid(True)

# #######
    ax02 = fig.add_subplot(spec[0, 5:])
    ax02.set_title('Generated Sequence. ($C_0; S_0$)')
    ax02.plot(generated_viz[0])
    ax02.grid(True) 

    ax12 = fig.add_subplot(spec[1, 5:])
    ax12.set_title('Generated Sequence. ($C_0; S_1$)')
    ax12.plot(generated_viz[1])
    ax12.grid(True) 
    
    
    ax10 = fig.add_subplot(spec[2, :4])
    ax10.set_title('Content Space.')
    
    ax11 = fig.add_subplot(spec[2, 4:])
    ax11.set_title('Style Space, Reduced with PCA.')   
     
    for i in range(n_style):
        draw_content_space(ax10, content_of_generated_viz[i], color=colors[i], label=f"content space style {i}")
        
        ax11.scatter(
            reduced_style[i, :, 0], 
            reduced_style[i,:, 1], 
            label=f'Real Style {i}.', 
            alpha=0.25
            )
        
        ax11.scatter(
            reduced_style_of_generated_viz[i, :, 0], 
            reduced_style_of_generated_viz[i,:, 1], 
            label=f'Generated Style {i}.', alpha=0.25
            )
        
        
        
    ax10.legend()
    ax11.legend()
    
    ax10.grid()    
    ax11.grid()
    
    plt.tight_layout()
    
    return fig