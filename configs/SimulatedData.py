from configs.Metric import MetricSimulatedData

class Proposed:
    sampling_period = 5 # Sampling period in minutes
    smoothing_period = 1*60 # Sampling period in minutes
    cols_on_interrest = ['in_c1', 'in_c2', 'out_c1', 'out_c2', 'out_c3', 'out_c4']
    
    use_mean_scaling = False

    sequence_lenght_in_sample = 64
    granularity = 1
    overlap= 0.25
    epochs = 50


    n_feature = len(cols_on_interrest)
    seq_shape = (sequence_lenght_in_sample, n_feature)
    batch_size = 20

    train_split = 0.7
    test_split  = 0.3
    valid_split = 0.2 

    reduce_train_set = False
    valid_set_batch_size= 50

    met_params = MetricSimulatedData()
        