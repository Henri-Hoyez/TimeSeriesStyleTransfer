from configs.SimulatedData import Proposed


class AmplitudeShiftArgs():
    def __init__(self) -> None:
        self.content_dataset_path = "data/simulated_dataset/01 - Source Domain.h5"
        self.style1_dataset_path = "data/simulated_dataset/output_noise/1.25.h5"
        self.style2_dataset_path = "data/simulated_dataset/output_noise/2.50.h5"

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "exp_folder"
        self.exp_name = "MTS ST V1 Noise"

        self.simulated_arguments = Proposed()



