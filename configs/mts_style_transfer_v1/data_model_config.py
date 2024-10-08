def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    # Stolen from the RainCOAT repository :P
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class OutputNoise():
    def __init__(self) -> None:
        pass



class TimeShift():
    def __init__(self) -> None:
        pass



class AmplitudeShift():
    def __init__(self) -> None:
        pass