class RealData():
    
    sampling_period = 15 # Sampling period in minutes
    sequence_lenght_in_sample = 52
    overlap= 1

    smoothing_period = 1*60 # Sampling period in minutes
    cols_on_interrest = [
        'PCI global flow rate operative set point', 
        'OLMB cycle dry coke rate - weight SP', 
        'Hot metal temperature (per sample)'
        ]
    
    use_mean_scaling = True
    
    sequence_duration = int(sampling_period* sequence_lenght_in_sample)
    
    shift_between_sequences = int(16)
    n_feature = len(cols_on_interrest)
    seq_shape = (sequence_lenght_in_sample, n_feature)
    batch_size = 16
    
    train_split = 0.7
    test_split  = 0.3
    valid_split = 0.2 

    reset_index= True


class Diligen(RealData):
    def __init__(self) -> None:
        super().__init__()
        self.dh4_cog_path = "data/Diligen/2023-09-29 - DH4_COG_process_filtered.h5"
        self.dh4_no_cog_path="data/Diligen/2023-09-29 - DH4_no_COG_process_filtered.h5"

        self.dh5_cog_path= "data/Diligen/2023-09-29 - DH5_with_COG_filtered.h5"
        self.dh5_no_cog_path="data/Diligen/2023-09-29 - DH5_no_COG_process_filtered.h5"


class Kardemir(RealData):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Kardemir Class have to be implemented.")
    

class Ternium(RealData):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Ternium Class have to be implemented.")


class Metric():
    sampling_period = 15
    smoothing_period= 1*60

    mean_senssitivity_factor= 10
    noise_senssitivity_factor=0.5

    signature_lenght=32