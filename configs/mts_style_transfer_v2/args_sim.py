from configs.SimulatedDataMultipleStyle import Proposed
from configs.RealData import PAMAP2
import json
class DafaultArguments():
    def __init__(self) -> None:
        self.content_dataset_path = "data/simulated_dataset/01 - Source Domain.h5"
        
        # self.style_datasets_path = [
        #     "data/simulated_dataset/amplitude_shift/1.0_1.0.h5", 
        #     "data/simulated_dataset/amplitude_shift/2.0_2.0.h5", 
        #     "data/simulated_dataset/amplitude_shift/3.0_3.0.h5", 
        #     "data/simulated_dataset/amplitude_shift/4.0_4.0.h5", 
        #     "data/simulated_dataset/amplitude_shift/5.0_5.0.h5", 
        #     "data/simulated_dataset/amplitude_shift/6.0_6.0.h5", 
        #     "data/simulated_dataset/amplitude_shift/7.0_7.0.h5" , 
        #     "data/simulated_dataset/amplitude_shift/8.0_8.0.h5" , 
        #     "data/simulated_dataset/amplitude_shift/9.0_9.0.h5" , 
        #     "data/simulated_dataset/amplitude_shift/10.0_10.0.h5"
        #     ]
        
        # self.style_datasets_path = [
        #     "data/simulated_dataset/amplitude_shift/1.0_1.0_standardized.h5",
        #     "data/simulated_dataset/amplitude_shift/2.0_2.0_standardized.h5",
        #     "data/simulated_dataset/amplitude_shift/3.0_3.0_standardized.h5",
        #     "data/simulated_dataset/amplitude_shift/4.0_4.0_standardized.h5",
        #     "data/simulated_dataset/amplitude_shift/5.0_5.0_standardized.h5",
        #     "data/simulated_dataset/amplitude_shift/6.0_6.0_standardized.h5",
        #     "data/simulated_dataset/amplitude_shift/7.0_7.0_standardized.h5" ,
        #     "data/simulated_dataset/amplitude_shift/8.0_8.0_standardized.h5" ,
        #     "data/simulated_dataset/amplitude_shift/9.0_9.0_standardized.h5" ,
        #     "data/simulated_dataset/amplitude_shift/10.0_10.0_standardized.h5",
        # ]
        
        self.style_datasets_path = [
            "data/simulated_dataset/output_noise/0.25.h5",
            "data/simulated_dataset/output_noise/0.50.h5",
            "data/simulated_dataset/output_noise/0.75.h5",
            "data/simulated_dataset/output_noise/1.00.h5",
            "data/simulated_dataset/output_noise/1.25.h5",
            "data/simulated_dataset/output_noise/1.50.h5",
            "data/simulated_dataset/output_noise/1.75.h5",
            "data/simulated_dataset/output_noise/2.00.h5",
            "data/simulated_dataset/output_noise/2.25.h5",
            "data/simulated_dataset/output_noise/2.50.h5"
        ]
        
        # self.style_datasets_path = [
        #     "data/simulated_dataset/time_shift/0.h5",
        #     "data/simulated_dataset/time_shift/2.h5",
        #     "data/simulated_dataset/time_shift/4.h5",
        #     "data/simulated_dataset/time_shift/6.h5",
        #     "data/simulated_dataset/time_shift/8.h5",
        #     "data/simulated_dataset/time_shift/10.h5",
        #     "data/simulated_dataset/time_shift/12.h5",
        #     "data/simulated_dataset/time_shift/14.h5",
        #     "data/simulated_dataset/time_shift/16.h5",
        #     "data/simulated_dataset/time_shift/18.h5"
        # ]

        self.simulated_arguments = Proposed()

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "page_blanche"
        self.exp_name = "Baseline"
        self.note = "Données standardizée, modèles plus petit."
        
        