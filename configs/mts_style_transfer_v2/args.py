from configs.SimulatedData import Proposed

class DafaultArguments():
    def __init__(self) -> None:
        self.content_dataset_path = "data/simulated_dataset/01 - Source Domain.h5"
        self.style_datasets_path = ["data/simulated_dataset/amplitude_shift/5.0_5.0.h5", "data/simulated_dataset/amplitude_shift/10.0_10.0.h5"]

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "exp_folder"
        self.exp_name = "MTS-ST V2 Amplitude Shift"

        self.simulated_arguments = Proposed()