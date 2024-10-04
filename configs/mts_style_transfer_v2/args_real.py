from configs.SimulatedDataMultipleStyle import Proposed
from configs.RealData import PAMAP2

class DafaultArguments():
    def __init__(self) -> None:    

        self.content_dataset_path = "data/PAMAP2/subject101.h5"

        self.style_datasets_path = [
            "data/PAMAP2/subject105.h5",
            "data/PAMAP2/subject106.h5",
            "data/PAMAP2/subject108.h5"
        ]
        
        self.simulated_arguments = PAMAP2()

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "exp_folder"
        self.exp_name = "MTS-ST V2 TS Real Data powered"