from configs.SimulatedDataMultipleStyle import Proposed
from configs.RealData import PAMAP2
import json
class DafaultArguments():
    def __init__(self) -> None:
        self.content_dataset_path = "data/simulated_dataset/01 - Source Domain_standardized.h5"
        
        self.style_datasets_path = [
            "data/simulated_dataset/output_noise/0.25_standardized.h5",
            "data/simulated_dataset/output_noise/0.50_standardized.h5",
            "data/simulated_dataset/output_noise/0.75_standardized.h5",
            "data/simulated_dataset/output_noise/1.00_standardized.h5",
            "data/simulated_dataset/output_noise/1.25_standardized.h5",
            "data/simulated_dataset/output_noise/1.50_standardized.h5",
            "data/simulated_dataset/output_noise/1.75_standardized.h5",
            "data/simulated_dataset/output_noise/2.00_standardized.h5",
            "data/simulated_dataset/output_noise/2.25_standardized.h5",
            "data/simulated_dataset/output_noise/2.50_standardized.h5"
        ]

        self.simulated_arguments = Proposed()

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "page_blanche"
        self.exp_name = "Baseline"
        self.note = "Données standardizée, modèles plus petit."
        
        