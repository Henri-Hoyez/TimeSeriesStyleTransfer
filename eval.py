import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

from configs.mts_style_transfer_v1.args import AmplitudeShiftArgs as args
from utils import simple_metric, eval_methods

from models.evaluation.utils import *

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.gpu_memory_grow import gpu_memory_grow

gpus = tf.config.list_physical_devices('GPU')
gpu_memory_grow(gpus)


def time_shift_evaluation(big_batch):
    return [simple_metric.estimate_time_shift(big_batch, 0, i) for i in range(big_batch.shape[-1])]


def viz_without_signatures(
        content_encoder, style_encoder, decoder, 
        content_sequences, 
        style1_sequences, 
        style2_sequences, root,
        title_extanssion=""):
    
    save_to = f"{root}/Sequences.png"

    # Make Generated sequence for visualization.
    content_of_content = content_encoder(content_sequences, training=False)
    style_of_style1= style_encoder(style1_sequences, training=False)
    style1_generated = decoder([content_of_content, style_of_style1], training=False)
    style1_generated = tf.concat(style1_generated, -1)

    style_of_style2 = style_encoder(style2_sequences, training=False)
    style2_generated = decoder([content_of_content, style_of_style2], training=False)
    style2_generated = tf.concat(style2_generated, -1)

    c_style1_generated = content_encoder(style1_generated, training=False)
    s_style1_generated = style_encoder(style1_generated, training=False)

    c_style2_generated = content_encoder(style2_generated, training=False)
    s_style2_generated = style_encoder(style2_generated, training=False)


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

    fig= plt.figure(figsize=(18, 10))
    fig.suptitle(f"Sequence Generations. {title_extanssion}")
    spec= fig.add_gridspec(3, 6)


    ax00 = fig.add_subplot(spec[:2, :2])
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

    ax11 =  fig.add_subplot(spec[1, 2:4])
    ax11.set_title("Style Sequence 2. ($S_1$)")
    ax11.plot(style2_sequences[0])
    ax11.set_ylim(_min, _max)
    ax11.grid(True)

# #######
    ax02 = fig.add_subplot(spec[0, 4:])
    ax02.set_title('Generated Sequence. ($C_0; S_0$)')
    ax02.plot(style1_generated[0])
    ax02.set_ylim(_min, _max)
    ax02.grid(True) 

    ax12 = fig.add_subplot(spec[1, 4:])
    ax12.set_title('Generated Sequence. ($C_0; S_1$)')
    ax12.plot(style2_generated[1])
    ax12.set_ylim(_min, _max)
    ax12.grid(True) 


# #####
    ax10 = fig.add_subplot(spec[2, 0:3])
    ax10.set_title('Content Space.')
    ax10.scatter(content_of_content[0, :, 0], content_of_content[0, :, 1],  label='Content of content.')
    ax10.scatter(c_style1_generated[0, :, 0], c_style1_generated[0, :, 1], label='Content of Generated style 1')
    ax10.scatter(c_style2_generated[0, :, 0], c_style2_generated[0, :, 1],  label='Content of Generated style 2')
    ax10.grid(True)
    ax10.legend()

    ax11 = fig.add_subplot(spec[2, 3:])
    ax11.set_title('Style Space, Reduced with PCA.')
    ax11.scatter(reduced_style1[:, 0], reduced_style1[:, 1], label='Style 1.', alpha=0.25)
    ax11.scatter(reduced_style2[:, 0], reduced_style2[:, 1], label='Style 2.', alpha=0.25)
    ax11.scatter(reduced_style1_generated[:, 0], reduced_style1_generated[:, 1], label='Generations style 1.', alpha=0.25)
    ax11.scatter(reduced_style2_generated[:, 0], reduced_style2_generated[:, 1], label='Generations style 2.', alpha=0.25)

    ax11.grid(True)
    ax11.legend()

    plt.tight_layout()
    fig.savefig(save_to)




def evaluate():
    default_params = args()

    shell_arguments = parse_arguments()

    root = f"{default_params.default_root_save_folder}/{shell_arguments.exp_folder}/{shell_arguments.exp_name}"

    trained_model_args = get_model_training_arguments(root)
    content_encoder, style_encoder, decoder = load_models(root)

    print("[+] Train Synthetic, Test on Real.")

    style1_r_perf, style1_g_perf = tstr(trained_model_args["dset_content"], trained_model_args['dset_style_1'], content_encoder, style_encoder, decoder, default_params)
    style2_r_perf, style2_g_perf = tstr(trained_model_args["dset_content"], trained_model_args['dset_style_2'], content_encoder, style_encoder, decoder, default_params)

    print('[+] Done.')
    print('[+] Classification on content space.')

    perf_content, perf_content_style1, perf_content_style2 = predictions_on_content_space(content_encoder, trained_model_args, default_params)

    print('[+] Done.')
    print('[+] Classification on Style space.')

    style_1_classif_perf = classification_on_style_space(trained_model_args['dset_style_1'], style_encoder, default_params)
    style_2_classif_perf = classification_on_style_space(trained_model_args['dset_style_2'], style_encoder, default_params)

    print('[+] Done.')
    print('[+] Real Fake Classification.')

    # Real / Fake Classification.
    style1_real_fake_classif = real_fake_classification(trained_model_args['dset_content'], trained_model_args['dset_style_1'], content_encoder, style_encoder, decoder, default_params)
    style2_real_fake_classif = real_fake_classification(trained_model_args['dset_content'], trained_model_args['dset_style_2'], content_encoder, style_encoder, decoder, default_params)

    print('[+] Done.')

    content_big_batch, style1_big_batch, style2_big_batch = load_valid_batches(trained_model_args)
    generated_style1 = generate(content_big_batch, style1_big_batch, content_encoder, style_encoder, decoder)
    generated_style2 = generate(content_big_batch, style2_big_batch, content_encoder, style_encoder, decoder)

    print("[+] Umap Visualization.")
    eval_methods.umap_plot(style1_big_batch[:500], style2_big_batch[:500], generated_style1[:500], generated_style2[:500], root, shell_arguments.exp_name)

    print("[+] t-SNE Visualization.")
    eval_methods.tsne_plot(style1_big_batch[:500], style2_big_batch[:500], generated_style1[:500], generated_style2[:500], root, shell_arguments.exp_name)


    print('[+] Plot Sequences.')
    viz_without_signatures(content_encoder, style_encoder, decoder, content_big_batch[:500], style1_big_batch[:500], style2_big_batch[:500], root, shell_arguments.exp_name)

    # plot_content_space(content_encoder, content_big_batch, style1_big_batch, style2_big_batch)

    print('[+] Simple Noise Metric.')
    _, content_extracted_noise = simple_metric.simple_metric_on_noise(content_big_batch)

    _, style1_extracted_noise = simple_metric.simple_metric_on_noise(style1_big_batch)
    _, style2_extracted_noise = simple_metric.simple_metric_on_noise(style2_big_batch)

    _, gen_s1_extracted_noise = simple_metric.simple_metric_on_noise(generated_style1)
    _, gen_s2_extracted_noise = simple_metric.simple_metric_on_noise(generated_style2)

    print('[+] Simple Amplitude Metric.')
    content_extracted_ampl = simple_metric.extract_amplitude_from_signals(content_big_batch)

    style1_extracted_ampl = simple_metric.extract_amplitude_from_signals(style1_big_batch)
    gen_s1_extracted_ampl = simple_metric.extract_amplitude_from_signals(generated_style1)

    style2_extracted_ampl = simple_metric.extract_amplitude_from_signals(style2_big_batch)
    
    gen_s2_extracted_ampl = simple_metric.extract_amplitude_from_signals(generated_style2)

    print("[+] Time shift Simple Metric.")

    content_shifts = time_shift_evaluation(content_big_batch)

    real_s1_shifts = time_shift_evaluation(style1_big_batch)
    real_s2_shifts = time_shift_evaluation(style2_big_batch)
    
    fake_s1_shifts = time_shift_evaluation(generated_style1)
    fake_s2_shifts = time_shift_evaluation(generated_style2)


    mbp_cols = [ 
        "style1_r_perf", "style1_g_perf", "style2_r_perf", "style2_g_perf", 
        "perf_content", "perf_content_style1", "perf_content_style2", 
        "style_1_classif_perf", "style_2_classif_perf", 
        "style1_real_fake_classif", "style2_real_fake_classif"
        ]

    mbp_values= [[ 
        style1_r_perf, style1_g_perf, style2_r_perf, style2_g_perf, 
        perf_content, perf_content_style1, perf_content_style2, 
        style_1_classif_perf, style_2_classif_perf, 
        style1_real_fake_classif, style2_real_fake_classif
        ]]
    
    smr_results = {
        "content_extracted_noise":content_extracted_noise,
        "style1_extracted_noise":style1_extracted_noise,
        "style2_extracted_noise":style2_extracted_noise,
        "gen_s1_extracted_noise":gen_s1_extracted_noise,
        "gen_s2_extracted_noise":gen_s2_extracted_noise,
        'content_extracted_ampl': content_extracted_ampl,
        'style1_extracted_ampl':style1_extracted_ampl,
        'style2_extracted_ampl':style2_extracted_ampl,
        'gen_s1_extracted_ampl':gen_s1_extracted_ampl,
        'gen_s2_extracted_ampl':gen_s2_extracted_ampl,
        "content_shifts":content_shifts,
        "real_s1_shifts":real_s1_shifts,
        "real_s2_shifts":real_s2_shifts,
        "fake_s1_shifts":fake_s1_shifts,
        "fake_s2_shifts":fake_s2_shifts
    }
    

    print(f'[+] Saving to the folder {root}... ')

    df = pd.DataFrame().from_dict(smr_results)
    df.to_excel(f"{root}/simple_metric_results.xlsx")

    df = pd.DataFrame(mbp_values, columns=mbp_cols)
    df.to_excel(f"{root}/model_based_predictions.xlsx")    

if __name__ == "__main__":
    evaluate()