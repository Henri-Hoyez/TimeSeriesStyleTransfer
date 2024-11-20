from configs.Metric import MetricSimulatedData

class Proposed:
    sampling_period = 5 # Sampling period in minutes
    smoothing_period = 1*60 # Sampling period in minutes
    cols_on_interrest = ['in_c1', 'in_c2', 'out_c1', 'out_c2', 'out_c3', 'out_c4']
    
    use_mean_scaling = False

    sequence_lenght_in_sample = 64
    granularity = 1
    overlap= 0.05
    epochs = 100

    n_feature = 7
    n_classes = 5
    seq_shape = (sequence_lenght_in_sample, n_feature)
    batch_size = 128

    train_split = 0.7
    test_split  = 0.3
    valid_split = 0.2 

    reduce_train_set = False
    valid_set_batch_size= 128

    # loss Parameters:
    n_styles = 2
    style_vector_size = 16
    
    n_wiener = 2
    n_sample_wiener = 8 #sequence_lenght_in_sample//4
    
    noise_dim = (n_sample_wiener, n_wiener)
    n_validation_sequences = 500
    discrinator_step = 1

    ##### Generator loss parameters.
    l_reconstr = 10
    l_local =  10
    l_global = 10
    l_style_preservation = 2.

    ##### Content encoder loss
    l_content = 2.
    encoder_adv = 0.2

    ##### Style Encoder
    l_disentanglement = 50.
    l_triplet = 50.
    triplet_r = 0.0
    
    # Train the generator or the discriminator based on the 
    # Performance of the Discriminator (Here the accuracy.)
    discriminator_success_threashold = 0.85
    alpha = 0.001
    normal_training_epochs = 0

    met_params = MetricSimulatedData()
        