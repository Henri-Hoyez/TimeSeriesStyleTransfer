class Metric():
    mean_senssibility_factor = 10
    noise_senssitivity = 0.5 


class MetricRealData(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.signature_length = 43200 # Two days with a Sampling period of 2 mins.


class MetricSimulatedData(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.signature_length = 32
        self.sequence_to_generate = 20

        self.ins = [0, 1]
        self.outs =[2, 3, 4, 5]