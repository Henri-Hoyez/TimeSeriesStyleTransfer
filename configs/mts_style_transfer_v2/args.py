from configs.SimulatedData import Proposed

class DafaultArguments():
    def __init__(self) -> None:
        self.content_dataset_path = "data/simulated_dataset/01 - Source Domain.h5"
        self.style_datasets_path = ["data/simulated_dataset/output_noise/1.25.h5", "data/simulated_dataset/output_noise/2.50.h5"]

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "exp_folder"
        self.exp_name = "MTS Style Transfer Amplitude Shift"

        self.simulated_arguments = Proposed()