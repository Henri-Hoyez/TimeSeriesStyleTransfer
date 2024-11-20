from configs.SimulatedDataMultipleStyle import Proposed
from configs.RealData import PAMAP2
import json
class DafaultArguments():
    def __init__(self) -> None:
        self.content_dataset_path = "data/simulated_dataset/01 - Source Domain_standardized.h5"
        
        self.style_datasets_path = [
            "data/simulated_dataset/output_noise/0.03_standardized.h5",
            "data/simulated_dataset/output_noise/0.05_standardized.h5",
            "data/simulated_dataset/output_noise/0.08_standardized.h5",
            "data/simulated_dataset/output_noise/0.10_standardized.h5",
            "data/simulated_dataset/output_noise/0.12_standardized.h5",
            "data/simulated_dataset/output_noise/0.15_standardized.h5",
            "data/simulated_dataset/output_noise/0.18_standardized.h5",
            "data/simulated_dataset/output_noise/0.20_standardized.h5",
            "data/simulated_dataset/output_noise/0.23_standardized.h5",
            "data/simulated_dataset/output_noise/0.25_standardized.h5",
        ]

        self.simulated_arguments = Proposed()

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "page_blanche"
        self.exp_name = "Baseline"
        self.note = "Données standardizée, modèles plus petit."
        
        