from configs.Metric import MetricSimulatedData

class Proposed:
    sampling_period = 5 # Sampling period in minutes
    smoothing_period = 1*60 # Sampling period in minutes
    cols_on_interrest = ['in_c1', 'in_c2', 'out_c1', 'out_c2', 'out_c3', 'out_c4']
    
    use_mean_scaling = False

    sequence_lenght_in_sample = 64
    granularity = 1
    overlap= 0.25
    epochs = 40

    n_feature = 7
    seq_shape = (sequence_lenght_in_sample, n_feature)
    batch_size = 20

    train_split = 0.7
    test_split  = 0.3
    valid_split = 0.2 

    reduce_train_set = False
    valid_set_batch_size= 50

    # loss Parameters:
    n_styles = 2
    style_vector_size = 16
    n_wiener = 2
    n_sample_wiener = 16 #sequence_lenght_in_sample//4
    noise_dim = (n_sample_wiener, n_wiener)
    n_validation_sequences = 500
    discrinator_step = 5

    ##### Generator loss parameters.
    l_reconstr = .1
    l_local = .1
    l_global = .1
    l_style_preservation = .45

    ##### Content encoder loss
    l_content = .1

    ##### Style Encoder
    l_disentanglement = .5
    triplet_r = .5
    l_triplet = .5

    met_params = MetricSimulatedData()
        