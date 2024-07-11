import numpy as np
from utils.metric import signature_on_batch
from configs.SimulatedData import Proposed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

config = Proposed()
def plot_generated_sequence(
        content_encoder, style_encoder, decoder, 
        content_sequences, 
        style1_sequences, 
        style2_sequences,
        config = config, 
        show=False):

    # Make Generated sequence for visualization.
    content_of_content = content_encoder(content_sequences, training=False)
    style_of_style1= style_encoder(style1_sequences, training=False)
    style1_generated = decoder([content_of_content, style_of_style1], training=False)

    style_of_style2 = style_encoder(style2_sequences, training=False)
    style2_generated = decoder([content_of_content, style_of_style2], training=False)

    c_style1_generated = content_encoder(style1_generated, training=False)
    s_style1_generated = style_encoder(style1_generated, training=False)

    c_style2_generated = content_encoder(style2_generated, training=False)
    s_style2_generated = style_encoder(style2_generated, training=False)

    style1_signature = signature_on_batch(style1_sequences, config.met_params.ins, config.met_params.outs, config.met_params.signature_length)
    style2_signature = signature_on_batch(style2_sequences, config.met_params.ins, config.met_params.outs, config.met_params.signature_length)
    gen_s1_signature = signature_on_batch(style1_generated, config.met_params.ins, config.met_params.outs, config.met_params.signature_length)
    gen_s2_signature = signature_on_batch(style2_generated, config.met_params.ins, config.met_params.outs, config.met_params.signature_length)

    x = np.arange(style1_signature.shape[1]) - style1_signature.shape[1]/2

    averaged_rs1_sig = np.mean(style1_signature, axis=0)
    averaged_rs2_sig = np.mean(style2_signature, axis=0)
    averaged_gs1_sig = np.mean(gen_s1_signature, axis=0)
    averaged_gs2_sig = np.mean(gen_s2_signature, axis=0)

    # Reduce the Style Vector for visualization purposes.
    pca = PCA(2)
    style_vectors = np.vstack(
        [   style_of_style1, 
            style_of_style2, 
            s_style1_generated, 
            s_style2_generated
        ])

    pca.fit(style_vectors)

    reduced_style1 = pca.transform(style_of_style1)
    reduced_style2 = pca.transform(style_of_style2)
    reduced_style1_generated = pca.transform(s_style1_generated)
    reduced_style2_generated = pca.transform(s_style2_generated)


    all_values = np.array([content_sequences, style1_sequences, style2_sequences])
    _min, _max = np.min(all_values)-1, np.max(all_values)+ 1

    fig= plt.figure(figsize=(18, 8))
    spec= fig.add_gridspec(3, 8)

    ax00 = fig.add_subplot(spec[0:2, :2])
    ax00.set_title('Content Sequence. ($C_0$)')
    ax00.plot(content_sequences[0])
    ax00.set_ylim(_min, _max)
    ax00.grid(True)
    ax00.legend()

# #######
    ax01 = fig.add_subplot(spec[0, 2:4])
    ax01.set_title('Style Sequence 1. ($S_0$)')
    ax01.plot(style1_sequences[0])
    ax01.set_ylim(_min, _max)
    ax01.grid(True)

    ax11 = fig.add_subplot(spec[1, 2:4])
    ax11.set_title("Style Sequence 2. ($S_1$)")
    ax11.plot(style2_sequences[0])
    ax11.set_ylim(_min, _max)
    ax11.grid(True)

# #######
    ax02 = fig.add_subplot(spec[0, 4:6])
    ax02.set_title('Generated Sequence. ($C_0; S_0$)')
    ax02.plot(style1_generated[0])
    ax02.set_ylim(_min, _max)
    ax02.grid(True) 

    ax12 = fig.add_subplot(spec[1, 4:6])
    ax12.set_title('Generated Sequence. ($C_0; S_1$)')
    ax12.plot(style2_generated[1])
    ax12.set_ylim(_min, _max)
    ax12.grid(True) 

# #####

    ax03 = fig.add_subplot(spec[0, 6:])
    ax03.set_title("Signature Style 1.")


    plt.plot(x, averaged_rs1_sig[:, 0], "g")
    plt.plot(x, averaged_rs1_sig[:, 1], "g")
    plt.plot(x, averaged_rs1_sig[:, 2], "g")

    plt.plot(x, averaged_gs1_sig[:, 0], "b")
    plt.plot(x, averaged_gs1_sig[:, 1], "b")
    plt.plot(x, averaged_gs1_sig[:, 2], "b")


    plt.fill_between(x, averaged_rs1_sig[:, 0].reshape((-1,)), averaged_rs1_sig[:, 1].reshape((-1,)), color="g", alpha=0.25, label="Signature From Real data, Style 1")
    plt.fill_between(x, averaged_gs1_sig[:, 0].reshape((-1,)), averaged_gs1_sig[:, 1].reshape((-1,)), color="b", alpha=0.25, label="Signature From Sim  data, Style 1")

    ax03.set_ylim(-1, 1)
    ax03.grid(True)
    ax03.legend()
    

    ax04 = fig.add_subplot(spec[1, 6:])
    ax04.set_title("Signature Style 2.")

    plt.plot(x, averaged_rs2_sig[:, 0], "g")
    plt.plot(x, averaged_rs2_sig[:, 1], "g")
    plt.plot(x, averaged_rs2_sig[:, 2], "g")

    plt.plot(x, averaged_gs2_sig[:, 0], "b")
    plt.plot(x, averaged_gs2_sig[:, 1], "b")
    plt.plot(x, averaged_gs2_sig[:, 2], "b")

    plt.fill_between(x, averaged_rs2_sig[:, 0].reshape((-1,)), averaged_rs2_sig[:, 1].reshape((-1,)), color="g", alpha=0.25, label="Signature From Real data, Style 2")
    plt.fill_between(x, averaged_gs2_sig[:, 0].reshape((-1,)), averaged_gs2_sig[:, 1].reshape((-1,)), color="b", alpha=0.25, label="Signature From Sim  data, Style 2")

    ax04.grid(True)
    ax04.set_ylim(-1, 1)
    ax04.legend()


# #####
    ax10 = fig.add_subplot(spec[2, :4])
    ax10.set_title('Content Space.')
    ax10.scatter(content_of_content[0, :, 0], content_of_content[0, :, 1],  label='Content of content.')
    ax10.scatter(c_style1_generated[0, :, 0], c_style1_generated[0, :, 1], label='Content of Generated style 1')
    ax10.scatter(c_style2_generated[0, :, 0], c_style2_generated[0, :, 1],  label='Content of Generated style 2')
    ax10.grid(True)
    ax10.legend()

    ax11 = fig.add_subplot(spec[2, 4:])
    ax11.set_title('Style Space, Reduced with PCA.')
    ax11.scatter(reduced_style1[:, 0], reduced_style1[:, 1], label='Style 1.', alpha=0.25)
    ax11.scatter(reduced_style2[:, 0], reduced_style2[:, 1], label='Style 2.', alpha=0.25)
    ax11.scatter(reduced_style1_generated[:, 0], reduced_style1_generated[:, 1], label='Generations style 1.', alpha=0.25)
    ax11.scatter(reduced_style2_generated[:, 0], reduced_style2_generated[:, 1], label='Generations style 2.', alpha=0.25)

    ax11.grid(True)
    ax11.legend()

    plt.tight_layout()

    return fig