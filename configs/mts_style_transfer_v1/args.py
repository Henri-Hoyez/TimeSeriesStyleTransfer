from configs.SimulatedData import Proposed


class AmplitudeShiftArgs():
    def __init__(self) -> None:
        self.content_dataset_path = "data/simulated_dataset/01 - Source Domain.h5"
        self.style1_dataset_path = "data/simulated_dataset/amplitude_shift/1.0_1.0.h5"
        self.style2_dataset_path = "data/simulated_dataset/amplitude_shift/4.5_4.5.h5"

        self.tensorboard_root_folder = "test2"
        self.default_root_save_folder = "to_evaluate"
        self.exp_name = "MTS Style Transfer Amplitude Shift"

        self.simulated_arguments = Proposed()



